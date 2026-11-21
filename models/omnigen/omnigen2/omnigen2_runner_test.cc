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
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/omnigen2_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "omni/omni_runner_lib.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

#ifndef __APPLE__
extern char** environ;
#endif

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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

TEST(OmniGen2RunnerTest, RunsExplicitModelNameAndWritesValidPng) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteOmniGen2Bundle("runner_explicit_model.litertlm"));
  const std::filesystem::path png_path =
      std::filesystem::path(::testing::TempDir()) / "runner_explicit.png";

  OmniRunnerOptions options;
  options.model_name = "omnigen2";
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "a cat on a mountain at sunset";
  options.steps = 2;
  options.seed = 42;
  options.output_path = png_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(out.str(), HasSubstr("Saved 32x32 image"));
  EXPECT_THAT(out.str(), HasSubstr(png_path.string()));

  ASSERT_OK_AND_ASSIGN(ParsedPngHeader png, ParsePngHeader(png_path));
  EXPECT_EQ(png.width, 32u);
  EXPECT_EQ(png.height, 32u);
  EXPECT_EQ(png.bit_depth, 8u);
  EXPECT_EQ(png.color_type, 2u);  // Truecolor RGB.
}

TEST(OmniGen2RunnerTest, InfersModelNameFromLitertLmFilename) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteOmniGen2Bundle("omnigen2_auto_infer.litertlm"));
  const std::filesystem::path png_path =
      std::filesystem::path(::testing::TempDir()) / "runner_inferred.png";

  OmniRunnerOptions options;
  options.model_path = bundle_path;
  options.cache_dir = ":nocache";
  options.num_threads = 1;
  options.text = "a robot painting";
  options.steps = 2;
  options.output_path = png_path.string();

  std::ostringstream out;
  ASSERT_OK(RunOmniRunner(options, out));
  EXPECT_THAT(out.str(), HasSubstr("Saved 32x32 image"));

  ASSERT_OK_AND_ASSIGN(ParsedPngHeader png, ParsePngHeader(png_path));
  EXPECT_EQ(png.width, 32u);
  EXPECT_EQ(png.height, 32u);
}

TEST(OmniGen2RunnerTest, RejectsMissingTextOrUnsupportedInputs) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteOmniGen2Bundle("omnigen2_reject_inputs.litertlm"));

  OmniRunnerOptions missing_text;
  missing_text.model_name = "omnigen2";
  missing_text.model_path = bundle_path;
  missing_text.cache_dir = ":nocache";
  std::ostringstream out;
  EXPECT_THAT(RunOmniRunner(missing_text, out),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniRunnerOptions with_audio = missing_text;
  with_audio.text = "a cat";
  with_audio.audio_path = "/tmp/unused.wav";
  EXPECT_THAT(RunOmniRunner(with_audio, out),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2RunnerTest, ExecutesOmniGen2RunnerBinaryEndToEnd) {
  const char* const workspace_env = std::getenv("TEST_WORKSPACE");
  const char* const workspace =
      (workspace_env != nullptr && workspace_env[0] != '\0') ? workspace_env
                                                             : "litert";
  const std::filesystem::path binary_path =
      std::filesystem::path(::testing::SrcDir()) / workspace /
      "models/omnigen/omnigen2/"
      "omnigen2_runner";
  ASSERT_TRUE(std::filesystem::exists(binary_path))
      << "Missing runner binary at: " << binary_path;

  // 1. Explicit `--model_name=omnigen2`.
  ASSERT_OK_AND_ASSIGN(
      std::string explicit_bundle,
      WriteOmniGen2Bundle("runner_cli_explicit_model.litertlm"));
  const std::filesystem::path explicit_png =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit.png";
  const std::filesystem::path explicit_log =
      std::filesystem::path(::testing::TempDir()) / "cli_explicit.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult explicit_res,
      RunSubprocess(binary_path,
                    {
                        "--model_name=omnigen2",
                        absl::StrCat("--model_path=", explicit_bundle),
                        "--text=a robot painting",
                        "--steps=2",
                        "--seed=42",
                        "--num_threads=1",
                        "--cache_dir=:nocache",
                        absl::StrCat("--output_path=", explicit_png.string()),
                    },
                    explicit_log));
  ASSERT_EQ(explicit_res.exit_code, 0) << explicit_res.combined_output;
  ASSERT_OK_AND_ASSIGN(ParsedPngHeader explicit_hdr,
                       ParsePngHeader(explicit_png));
  EXPECT_EQ(explicit_hdr.width, 32u);
  EXPECT_EQ(explicit_hdr.height, 32u);

  // 2. Filename-inferred model name without `--model_name`.
  ASSERT_OK_AND_ASSIGN(std::string inferred_bundle,
                       WriteOmniGen2Bundle("omnigen2_cli_inferred.litertlm"));
  const std::filesystem::path inferred_png =
      std::filesystem::path(::testing::TempDir()) / "cli_inferred.png";
  const std::filesystem::path inferred_log =
      std::filesystem::path(::testing::TempDir()) / "cli_inferred.log";

  ASSERT_OK_AND_ASSIGN(
      SubprocessResult inferred_res,
      RunSubprocess(binary_path,
                    {
                        absl::StrCat("--model_path=", inferred_bundle),
                        "--text=a cat",
                        "--steps=2",
                        "--num_threads=1",
                        "--cache_dir=:nocache",
                        absl::StrCat("--output_path=", inferred_png.string()),
                    },
                    inferred_log));
  ASSERT_EQ(inferred_res.exit_code, 0) << inferred_res.combined_output;
  ASSERT_OK_AND_ASSIGN(ParsedPngHeader inferred_hdr,
                       ParsePngHeader(inferred_png));
  EXPECT_EQ(inferred_hdr.width, 32u);
  EXPECT_EQ(inferred_hdr.height, 32u);
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2
