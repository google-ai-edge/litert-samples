// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <crt_externs.h>
#endif

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/lfm2_audio.pb.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "omni/omni_runner_lib.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

#ifndef __APPLE__
extern char** environ;
#endif

namespace litert::omni::any2any {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

struct ParsedPcmWav {
  uint16_t audio_format = 0;
  uint16_t num_channels = 0;
  uint32_t sample_rate_hz = 0;
  uint16_t bits_per_sample = 0;
  std::vector<int16_t> samples;
};

uint16_t ReadLe16(const char* p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(p[1])) << 8);
}

uint32_t ReadLe32(const char* p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

absl::StatusOr<ParsedPcmWav> ParsePcmWavFile(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(std::string bytes, ReadFileBytes(path));
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("File is not a valid RIFF/WAVE file: ", path.string()));
  }

  ParsedPcmWav wav;
  bool found_fmt = false;
  bool found_data = false;
  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const absl::string_view chunk_id =
        absl::string_view(bytes).substr(offset, 4);
    const uint32_t chunk_size = ReadLe32(bytes.data() + offset + 4);
    offset += 8;
    if (offset + chunk_size > bytes.size()) {
      return absl::InvalidArgumentError("Truncated WAV chunk.");
    }
    if (chunk_id == "fmt ") {
      if (chunk_size < 16) {
        return absl::InvalidArgumentError("WAV fmt chunk too small.");
      }
      wav.audio_format = ReadLe16(bytes.data() + offset);
      wav.num_channels = ReadLe16(bytes.data() + offset + 2);
      wav.sample_rate_hz = ReadLe32(bytes.data() + offset + 4);
      wav.bits_per_sample = ReadLe16(bytes.data() + offset + 14);
      found_fmt = true;
    } else if (chunk_id == "data") {
      if (chunk_size % sizeof(int16_t) != 0) {
        return absl::InvalidArgumentError(
            "WAV data chunk size is not a multiple of 2 bytes.");
      }
      const size_t num_samples = chunk_size / sizeof(int16_t);
      wav.samples.resize(num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        wav.samples[i] = static_cast<int16_t>(
            ReadLe16(bytes.data() + offset + i * sizeof(int16_t)));
      }
      found_data = true;
    }
    offset += chunk_size + (chunk_size & 1U);
  }
  if (!found_fmt || !found_data) {
    return absl::InvalidArgumentError("WAV file missing fmt or data chunk.");
  }
  return wav;
}

struct SubprocessResult {
  int exit_code = -1;
  std::string combined_output;
};

absl::StatusOr<SubprocessResult> RunSubprocess(
    const std::filesystem::path& binary_path,
    const std::vector<std::string>& args,
    const std::filesystem::path& log_path) {
  std::vector<std::string> argv_storage;
  argv_storage.reserve(args.size() + 1);
  argv_storage.push_back(binary_path.string());
  argv_storage.insert(argv_storage.end(), args.begin(), args.end());

  std::vector<char*> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string& arg : argv_storage) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions) != 0) {
    return absl::InternalError("posix_spawn_file_actions_init failed.");
  }
  absl::Cleanup destroy_actions = [&file_actions] {
    posix_spawn_file_actions_destroy(&file_actions);
  };

  const std::string log_path_str = log_path.string();
  if (posix_spawn_file_actions_addopen(
          &file_actions, STDOUT_FILENO, log_path_str.c_str(),
          O_WRONLY | O_CREAT | O_TRUNC, 0600) != 0 ||
      posix_spawn_file_actions_adddup2(&file_actions, STDOUT_FILENO,
                                       STDERR_FILENO) != 0) {
    return absl::InternalError("posix_spawn_file_actions setup failed.");
  }

#ifdef __APPLE__
  char** child_environ = *_NSGetEnviron();
#else
  char** child_environ = environ;
#endif

  pid_t pid = 0;
  const int spawn_err =
      posix_spawn(&pid, binary_path.c_str(), &file_actions, /*attrp=*/nullptr,
                  argv.data(), child_environ);
  if (spawn_err != 0) {
    return absl::InternalError(
        absl::StrCat("posix_spawn failed with errno ", spawn_err));
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      return absl::InternalError(
          absl::StrCat("waitpid failed with errno ", errno));
    }
  }

  SubprocessResult result;
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  if (auto log_bytes = ReadFileBytes(log_path); log_bytes.ok()) {
    result.combined_output = *std::move(log_bytes);
  }
  return result;
}

TEST(Lfm2AudioRunnerTest, RunsTextOnlyPromptAndWritesTranscriptFile) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_runner_text_only.litertlm"));
  const std::filesystem::path transcript_path =
      std::filesystem::path(::testing::TempDir()) / "runner_text_only.txt";

  OmniRunnerOptions options;
  options.model_name = "lfm2.5-audio-1.5b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "Hello";
  options.generate_audio = false;
  options.max_new_tokens = 2;
  options.temperature = 0.0f;
  options.output_path = transcript_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  const std::string trimmed_out(absl::StripAsciiWhitespace(out.str()));
  EXPECT_THAT(trimmed_out, Not(IsEmpty()));

  ASSERT_OK_AND_ASSIGN(std::string file_contents,
                       ReadFileBytes(transcript_path));
  EXPECT_EQ(absl::StripAsciiWhitespace(file_contents), trimmed_out);
}

TEST(Lfm2AudioRunnerTest, RunsAudioAndTextPromptWithRealWavFile) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_runner_audio_text.litertlm"));
  const std::filesystem::path wav_path =
      std::filesystem::path(::testing::TempDir()) / "runner_input_16k.wav";
  const std::filesystem::path transcript_path =
      std::filesystem::path(::testing::TempDir()) / "runner_audio_text.txt";

  std::vector<float> pcm(1600);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * 440.0f *
                             static_cast<float>(i) / 16000.0f);
  }
  ASSERT_OK(WriteTestWavFile(wav_path, pcm, /*sample_rate_hz=*/16000));

  OmniRunnerOptions options;
  options.model_name = "lfm2.5-audio-1.5b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.audio_path = wav_path.string();
  options.text = "Hello";
  options.generate_audio = false;
  options.max_new_tokens = 2;
  options.temperature = 0.0f;
  options.output_path = transcript_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(absl::StripAsciiWhitespace(out.str()), Not(IsEmpty()));

  ASSERT_OK_AND_ASSIGN(std::string file_contents,
                       ReadFileBytes(transcript_path));
  EXPECT_THAT(absl::StripAsciiWhitespace(file_contents), Not(IsEmpty()));
}

TEST(Lfm2AudioRunnerTest, InfersModelNameFromLitertLmFilename) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_audio_auto_infer.litertlm"));

  OmniRunnerOptions options;
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "Hello";
  options.generate_audio = false;
  options.max_new_tokens = 2;
  options.temperature = 0.0f;

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(absl::StripAsciiWhitespace(out.str()), Not(IsEmpty()));
}

TEST(Lfm2AudioRunnerTest, GeneratesSpeechAndWritesValid24kHzMonoPcmWav) {
  ASSERT_OK_AND_ASSIGN(lm::proto::Any2AnyMetadata meta,
                       BuildDefaultTestMetadata());
  prototypes::litert_models::lfm2_audio::Lfm2Audio lfm2;
  ASSERT_TRUE(meta.any2any_model_type().custom_model().UnpackTo(&lfm2));
  // Split the 32-element output of `checksum_fake_lm_embedding_input.tflite`
  // into 28 vocab logits + 4 hidden-state dimensions so speech generation can
  // extract `llm_hidden` without mocks.
  lfm2.set_llm_vocab_size(28);
  lfm2.set_llm_hidden_size(4);
  lfm2.set_tts_hidden_size(4);
  lfm2.set_tts_vocab_size(8);
  lfm2.set_tts_num_codebooks(8);
  lfm2.set_generate_audio(true);
  lfm2.set_max_new_tokens(8);
  lfm2.set_max_tts_tokens(4);
  ASSERT_TRUE(
      meta.mutable_any2any_model_type()->mutable_custom_model()->PackFrom(
          lfm2));

  Lfm2AudioTestBundleOptions bundle_options;
  bundle_options.custom_meta = &meta;
  bundle_options.include_tts_models = true;
  bundle_options.llm_hidden_size = 4;
  bundle_options.tts_hidden_size = 4;
  bundle_options.tts_vocab_size = 8;
  bundle_options.tts_num_codebooks = 8;
  bundle_options.vocoder_chunk_frames = 2;
  bundle_options.vocoder_samples_per_frame = 8;
  bundle_options.vocoder_output_amplitude = 0.25f;

  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteLfm2AudioBundle("lfm2_runner_speech.litertlm", bundle_options));
  const std::filesystem::path out_wav_path =
      std::filesystem::path(::testing::TempDir()) / "runner_speech_out.wav";

  OmniRunnerOptions options;
  options.model_name = "lfm2.5-audio-1.5b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "Hello";
  options.generate_audio = true;
  options.max_new_tokens = 8;
  options.temperature = 0.0f;
  options.output_path = out_wav_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(out.str(), HasSubstr("@ 24000 Hz"));
  EXPECT_THAT(out.str(), HasSubstr(out_wav_path.string()));

  ASSERT_OK_AND_ASSIGN(ParsedPcmWav wav, ParsePcmWavFile(out_wav_path));
  EXPECT_EQ(wav.audio_format, 1);
  EXPECT_EQ(wav.num_channels, 1);
  EXPECT_EQ(wav.sample_rate_hz, 24000);
  EXPECT_EQ(wav.bits_per_sample, 16);
  // Interleaved mode switches to audio after 6 text steps, generating 2 audio
  // frames * 8 samples/frame = 16 PCM samples.
  ASSERT_EQ(wav.samples.size(), 16u);
  const int16_t expected_pcm = static_cast<int16_t>(std::lround(0.25f * 32767));
  for (int16_t sample : wav.samples) {
    EXPECT_NEAR(sample, expected_pcm, 2);
  }
}

TEST(Lfm2AudioRunnerTest, ExecutesLfm2AudioRunnerBinaryEndToEnd) {
  const char* const workspace_env = std::getenv("TEST_WORKSPACE");
  const char* const workspace =
      (workspace_env != nullptr && workspace_env[0] != '\0') ? workspace_env
                                                             : "litert";
  const std::filesystem::path binary_path =
      std::filesystem::path(::testing::SrcDir()) / workspace /
      "models/lfm/lfm2_5_audio/"
      "lfm2_audio_runner";
  ASSERT_TRUE(std::filesystem::exists(binary_path))
      << "Missing runner binary at: " << binary_path;

  // 1. Explicit `--model_name=lfm2.5-audio-1.5b`.
  ASSERT_OK_AND_ASSIGN(
      std::string explicit_bundle,
      WriteLfm2AudioBundle("runner_cli_explicit_model.litertlm"));
  const std::filesystem::path explicit_out =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit_out.txt";
  const std::filesystem::path explicit_log =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult explicit_res,
      RunSubprocess(binary_path,
                    {
                        "--model_name=lfm2.5-audio-1.5b",
                        absl::StrCat("--model_path=", explicit_bundle),
                        "--text=Hello",
                        "--generate_audio=false",
                        "--max_new_tokens=2",
                        "--temperature=0.0",
                        "--num_threads=1",
                        "--cache_dir=:nocache",
                        absl::StrCat("--output_path=", explicit_out.string()),
                    },
                    explicit_log));
  ASSERT_EQ(explicit_res.exit_code, 0) << explicit_res.combined_output;
  ASSERT_OK_AND_ASSIGN(std::string explicit_text, ReadFileBytes(explicit_out));
  EXPECT_THAT(absl::StripAsciiWhitespace(explicit_text), Not(IsEmpty()));

  // 2. Filename-inferred model name without `--model_name`.
  ASSERT_OK_AND_ASSIGN(
      std::string inferred_bundle,
      WriteLfm2AudioBundle("lfm2_audio_cli_inferred.litertlm"));
  const std::filesystem::path inferred_out =
      std::filesystem::path(::testing::TempDir()) / "cli_inferred_out.txt";
  const std::filesystem::path inferred_log =
      std::filesystem::path(::testing::TempDir()) / "cli_inferred.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult inferred_res,
      RunSubprocess(binary_path,
                    {
                        absl::StrCat("--model_path=", inferred_bundle),
                        "--text=Hello",
                        "--generate_audio=false",
                        "--max_new_tokens=2",
                        "--temperature=0.0",
                        "--num_threads=1",
                        "--cache_dir=:nocache",
                        absl::StrCat("--output_path=", inferred_out.string()),
                    },
                    inferred_log));
  ASSERT_EQ(inferred_res.exit_code, 0) << inferred_res.combined_output;
  ASSERT_OK_AND_ASSIGN(std::string inferred_text, ReadFileBytes(inferred_out));
  EXPECT_THAT(absl::StripAsciiWhitespace(inferred_text), Not(IsEmpty()));
}

}  // namespace
}  // namespace litert::omni::any2any
