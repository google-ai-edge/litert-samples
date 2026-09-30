// Copyright 2026 The AI Edge LiteRT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "BenchmarkRun.h"

#import <LiteRtBenchmark/litert_benchmark_shim.h>

#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

NSString* DocumentsDir() {
  NSArray<NSString*>* dirs =
      NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  return dirs.firstObject;
}

// benchmark_model flags whose value is a path; relative values resolve against Documents,
// so the caller can say --graph=model.tflite --result_file_path=results.pb.
const char* const kPathFlags[] = {"--graph=", "--result_file_path=",
                                  "--model_runtime_info_output_file=",
                                  "--profiling_output_csv_file="};

std::string ResolvePathFlag(const std::string& arg) {
  for (const char* flag : kPathFlags) {
    const size_t n = std::strlen(flag);
    if (arg.compare(0, n, flag) == 0 && arg.size() > n && arg[n] != '/') {
      NSString* value = @(arg.c_str() + n);
      return flag + std::string(LiteRtBenchmarkDocumentsPath(value).UTF8String);
    }
  }
  return arg;
}

// The runtime finds the GPU accelerator by file name (libLiteRtMetalAccelerator.dylib) with no
// directory when no runtime library dir is set. The dylib is embedded in the app's Frameworks
// folder: load it from its absolute path first, and make that folder the working directory so
// a name-only dlopen resolves there as well. Once per process.
void PrepareGpuAccelerator() {
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  NSString* frameworks = NSBundle.mainBundle.privateFrameworksPath;
  NSString* dylib = [frameworks stringByAppendingPathComponent:@"libLiteRtMetalAccelerator.dylib"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:dylib]) {
    return;
  }
  chdir(frameworks.UTF8String);
  void* handle = dlopen(dylib.UTF8String, RTLD_NOW | RTLD_GLOBAL);
  fprintf(stderr, "INFO: GPU accelerator dylib %s: %s\n", dylib.UTF8String,
          handle != nullptr ? "loaded" : dlerror());
}

// The tool's flags and log streams are process-wide: one run at a time.
std::atomic<bool> g_running{false};

}  // namespace

NSString* LiteRtBenchmarkDocumentsPath(NSString* name) {
  return [DocumentsDir() stringByAppendingPathComponent:name];
}

int LiteRtBenchmarkRun(NSArray<NSString*>* flags, NSString* logName) {
  if (g_running.exchange(true)) {
    fprintf(stderr, "ERROR: a benchmark_model run is already in progress in this process\n");
    return 1;
  }
  std::vector<std::string> args = {"benchmark_model"};
  for (NSString* flag in flags) {
    args.push_back(ResolvePathFlag(flag.UTF8String));
  }
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // benchmark_model writes its results to stderr and its progress (STARTING!, the run
  // schedule, the delegate lines) to stdout. For this run, both go through one pipe whose
  // reader copies everything into the log file and onto the original stdout (what
  // `devicectl ... --console` and xcodebuild bridge). Restoring the two descriptors
  // afterwards closes the pipe, which ends the reader.
  FILE* log = fopen(LiteRtBenchmarkDocumentsPath(logName).UTF8String, "w");
  const int saved_stdout = dup(STDOUT_FILENO);
  const int saved_stderr = dup(STDERR_FILENO);
  int fds[2];
  dispatch_group_t readers = dispatch_group_create();
  if (log != nullptr && saved_stdout >= 0 && saved_stderr >= 0 && pipe(fds) == 0) {
    const int reader = fds[0];
    fflush(stdout);
    fflush(stderr);
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    dispatch_group_async(readers, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
      char buf[4096];
      while (true) {
        const ssize_t n = read(reader, buf, sizeof(buf));
        if (n < 0 && errno == EINTR) {
          continue;
        }
        if (n <= 0) {
          break;
        }
        write(saved_stdout, buf, n);
        fwrite(buf, 1, n, log);
        fflush(log);
      }
      close(reader);
    });
  } else {
    fprintf(stderr, "WARNING: the log is not mirrored into %s\n", logName.UTF8String);
  }
  PrepareGpuAccelerator();
  const int status = LiteRtBenchmarkMain(static_cast<int>(argv.size()), argv.data());
  fflush(stdout);
  fflush(stderr);
  if (saved_stdout >= 0) {
    dup2(saved_stdout, STDOUT_FILENO);
  }
  if (saved_stderr >= 0) {
    dup2(saved_stderr, STDERR_FILENO);  // the pipe's last write end closes: the reader sees EOF
  }
  dispatch_group_wait(readers, DISPATCH_TIME_FOREVER);
  if (saved_stdout >= 0) {
    close(saved_stdout);
  }
  if (saved_stderr >= 0) {
    close(saved_stderr);
  }
  if (log != nullptr) {
    fclose(log);
  }
  g_running = false;
  return status;
}
