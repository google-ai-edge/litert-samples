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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/janus_pro_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "omni/omni_runner_lib.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

#ifndef __APPLE__
extern char** environ;
#endif

namespace litert::omni::any2any {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr unsigned char kPngSignature[8] = {0x89, 'P',  'N',  'G',
                                            '\r', '\n', 0x1A, '\n'};

absl::StatusOr<lm::DecodedImage> VerifyAndDecodePngFile(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(std::string bytes, ReadFileBytes(path));
  if (bytes.size() < 24 ||
      std::memcmp(bytes.data(), kPngSignature, sizeof(kPngSignature)) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("File is not a valid PNG file: ", path.string()));
  }
  std::unique_ptr<lm::ImagePreprocessor> preprocessor =
      lm::ImagePreprocessor::Create();
  if (preprocessor == nullptr) {
    return absl::InternalError("ImagePreprocessor::Create returned null.");
  }
  return preprocessor->Decode(bytes);
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

TEST(JanusProRunnerTest, RunsTextOnlyPromptAndWritesTranscriptFile) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteJanusProBundle("janus_runner_text_only.litertlm"));
  const std::filesystem::path transcript_path =
      std::filesystem::path(::testing::TempDir()) / "runner_text_only.txt";

  OmniRunnerOptions options;
  options.model_name = "janus-pro-1b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "Hello";
  options.generate_image = false;
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

TEST(JanusProRunnerTest, RunsImageAndTextPromptWithRealPngFile) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteJanusProBundle("janus_runner_image_text.litertlm"));
  const std::filesystem::path png_path =
      std::filesystem::path(::testing::TempDir()) / "runner_input.png";
  const std::filesystem::path transcript_path =
      std::filesystem::path(::testing::TempDir()) / "runner_image_text.txt";

  std::vector<uint8_t> pixels(8 * 8 * 3, 180);
  ASSERT_OK(WriteTestPngFile(png_path, pixels, /*width=*/8, /*height=*/8,
                             /*channels=*/3));

  OmniRunnerOptions options;
  options.model_name = "janus-pro-1b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.image_path = png_path.string();
  options.text = "Describe";
  options.generate_image = false;
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

TEST(JanusProRunnerTest, InfersModelNameFromLitertLmFilename) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteJanusProBundle("janus_pro_auto_infer.litertlm"));

  OmniRunnerOptions options;
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "Hello";
  options.generate_image = false;
  options.max_new_tokens = 2;
  options.temperature = 0.0f;

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(absl::StripAsciiWhitespace(out.str()), Not(IsEmpty()));
}

TEST(JanusProRunnerTest, GeneratesImageAndWritesValidPngFile) {
  JanusProTestBundleOptions bundle_opts;
  bundle_opts.llm_vocab_size = 28;
  bundle_opts.llm_hidden_size = 4;
  bundle_opts.vq_codebook_size = 8;
  bundle_opts.image_size = 16;
  bundle_opts.patch_size = 8;
  bundle_opts.num_image_tokens = 4;
  bundle_opts.generate_image = true;
  bundle_opts.cfg_weight = 5.0f;
  bundle_opts.image_temperature = 0.0f;
  bundle_opts.vq_decoder_r = 1.0f;
  bundle_opts.vq_decoder_g = 0.0f;
  bundle_opts.vq_decoder_b = -1.0f;

  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteJanusProBundle("janus_runner_image_gen.litertlm", bundle_opts));
  const std::filesystem::path out_png_path =
      std::filesystem::path(::testing::TempDir()) / "runner_image_gen_out.png";

  OmniRunnerOptions options;
  options.model_name = "janus-pro-1b";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "fox";
  options.generate_image = true;
  options.cfg_weight = 5.0f;
  options.image_temperature = 0.0f;
  options.seed = 42;
  options.output_path = out_png_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(out.str(), HasSubstr("Saved 16x16 image to"));
  EXPECT_THAT(out.str(), HasSubstr(out_png_path.string()));

  ASSERT_OK_AND_ASSIGN(lm::DecodedImage decoded,
                       VerifyAndDecodePngFile(out_png_path));
  EXPECT_EQ(decoded.width, 16);
  EXPECT_EQ(decoded.height, 16);
  ASSERT_EQ(decoded.pixels.size(), 16u * 16u * 3u);
  EXPECT_EQ(decoded.pixels[0], 255);
  EXPECT_EQ(decoded.pixels[1], 127);
  EXPECT_EQ(decoded.pixels[2], 0);
}

TEST(JanusProRunnerTest, ExecutesJanusProRunnerBinaryEndToEnd) {
  const char* const workspace_env = std::getenv("TEST_WORKSPACE");
  const char* const workspace =
      (workspace_env != nullptr && workspace_env[0] != '\0') ? workspace_env
                                                             : "litert";
  const std::filesystem::path binary_path =
      std::filesystem::path(::testing::SrcDir()) / workspace /
      "models/deepseek/janus_pro/runner/"
      "janus_pro_runner";
  ASSERT_TRUE(std::filesystem::exists(binary_path))
      << "Missing runner binary at: " << binary_path;

  // 1. Explicit `--model_name=janus-pro-1b` understanding run.
  ASSERT_OK_AND_ASSIGN(
      std::string explicit_bundle,
      WriteJanusProBundle("runner_cli_explicit_model.litertlm"));
  const std::filesystem::path explicit_out =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit_out.txt";
  const std::filesystem::path explicit_log =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult explicit_res,
      RunSubprocess(binary_path,
                    {
                        "--model_name=janus-pro-1b",
                        absl::StrCat("--model_path=", explicit_bundle),
                        "--text=Hello",
                        "--generate_image=false",
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
  ASSERT_OK_AND_ASSIGN(std::string inferred_bundle,
                       WriteJanusProBundle("janus_pro_cli_inferred.litertlm"));
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
                        "--generate_image=false",
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

  // 3. Visual generation CLI run with `--generate_image=true` writing a valid
  //    PNG.
  JanusProTestBundleOptions gen_opts;
  gen_opts.llm_vocab_size = 28;
  gen_opts.llm_hidden_size = 4;
  gen_opts.vq_codebook_size = 8;
  gen_opts.image_size = 16;
  gen_opts.patch_size = 8;
  gen_opts.num_image_tokens = 4;
  gen_opts.generate_image = true;
  gen_opts.vq_decoder_r = 1.0f;
  gen_opts.vq_decoder_g = 0.0f;
  gen_opts.vq_decoder_b = -1.0f;
  ASSERT_OK_AND_ASSIGN(
      std::string gen_bundle,
      WriteJanusProBundle("janus_pro_cli_gen.litertlm", gen_opts));
  const std::filesystem::path gen_png_out =
      std::filesystem::path(::testing::TempDir()) / "cli_gen_out.png";
  const std::filesystem::path gen_log =
      std::filesystem::path(::testing::TempDir()) / "cli_gen.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult gen_res,
      RunSubprocess(binary_path,
                    {
                        "--model_name=janus-pro-1b",
                        absl::StrCat("--model_path=", gen_bundle),
                        "--text=fox",
                        "--generate_image=true",
                        "--cfg_weight=5.0",
                        "--image_temperature=0.0",
                        "--seed=42",
                        "--num_threads=1",
                        "--cache_dir=:nocache",
                        absl::StrCat("--output_path=", gen_png_out.string()),
                    },
                    gen_log));
  ASSERT_EQ(gen_res.exit_code, 0) << gen_res.combined_output;
  ASSERT_OK_AND_ASSIGN(lm::DecodedImage decoded,
                       VerifyAndDecodePngFile(gen_png_out));
  EXPECT_EQ(decoded.width, 16);
  EXPECT_EQ(decoded.height, 16);
  ASSERT_EQ(decoded.pixels.size(), 16u * 16u * 3u);
  EXPECT_EQ(decoded.pixels[0], 255);
  EXPECT_EQ(decoded.pixels[1], 127);
  EXPECT_EQ(decoded.pixels[2], 0);
}

}  // namespace
}  // namespace litert::omni::any2any
