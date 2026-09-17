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

#import "AppDelegate.h"

#import <LiteRtBenchmark/litert_benchmark_shim.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

NSString* DocumentsDir() {
  return NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
}

NSString* DocumentsPath(NSString* name) {
  return [DocumentsDir() stringByAppendingPathComponent:name];
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
      return flag + std::string(DocumentsPath(value).UTF8String);
    }
  }
  return arg;
}

// argv for benchmark_model: the app's own launch arguments (anything starting with "--"),
// or, when there are none, the keys of Documents/benchmark_params.json as --key=value.
std::vector<std::string> BenchmarkArgs() {
  std::vector<std::string> args = {"benchmark_model"};
  NSArray<NSString*>* launch_args = NSProcessInfo.processInfo.arguments;
  for (NSUInteger i = 1; i < launch_args.count; ++i) {
    if ([launch_args[i] hasPrefix:@"--"]) args.push_back(launch_args[i].UTF8String);
  }
  if (args.size() == 1) {
    NSData* data = [NSData dataWithContentsOfFile:DocumentsPath(@"benchmark_params.json")];
    NSDictionary* params =
        data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    for (NSString* key in params) {
      args.push_back(std::string("--") + key.UTF8String + "=" +
                     [params[key] description].UTF8String);
    }
  }
  for (auto& arg : args) arg = ResolvePathFlag(arg);
  return args;
}

// benchmark_model logs to stderr. Mirror it into Documents/benchmark.log and echo it on stdout,
// which is what `devicectl ... --console` bridges, so the console and a later file pull see the same text.
void TeeStderrToFile(NSString* path) {
  int fds[2];
  if (pipe(fds) != 0) return;
  FILE* log = fopen(path.UTF8String, "w");
  dup2(fds[1], STDERR_FILENO);
  close(fds[1]);
  const int reader = fds[0];
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    char buf[4096];
    ssize_t n;
    while ((n = read(reader, buf, sizeof(buf))) > 0) {
      write(STDOUT_FILENO, buf, n);
      if (log != nullptr) {
        fwrite(buf, 1, n, log);
        fflush(log);
      }
    }
  });
}

// Launched from the command line (devicectl or an Xcode scheme) means a driver is waiting for the
// process to end; launched by tapping the icon means a person is reading the screen.
bool LaunchedWithFlags() {
  NSArray<NSString*>* launch_args = NSProcessInfo.processInfo.arguments;
  for (NSUInteger i = 1; i < launch_args.count; ++i) {
    if ([launch_args[i] hasPrefix:@"--"]) return true;
  }
  return false;
}

// The runtime finds the GPU accelerator by file name (libLiteRtMetalAccelerator.dylib) with no
// directory when no runtime library dir is set. The dylib is embedded in the app's Frameworks
// folder: load it from its absolute path first, and make that folder the working directory so
// a name-only dlopen resolves there as well.
void PrepareGpuAccelerator() {
  NSString* frameworks = NSBundle.mainBundle.privateFrameworksPath;
  NSString* dylib = [frameworks stringByAppendingPathComponent:@"libLiteRtMetalAccelerator.dylib"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:dylib]) return;
  chdir(frameworks.UTF8String);
  void* handle = dlopen(dylib.UTF8String, RTLD_NOW | RTLD_GLOBAL);
  fprintf(stderr, "INFO: GPU accelerator dylib %s: %s\n", dylib.UTF8String,
          handle != nullptr ? "loaded" : dlerror());
}

}  // namespace

@implementation AppDelegate

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  application.idleTimerDisabled = YES;
  [[NSFileManager defaultManager] removeItemAtPath:DocumentsPath(@"benchmark.done") error:nil];
  TeeStderrToFile(DocumentsPath(@"benchmark.log"));

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    std::vector<std::string> args = BenchmarkArgs();
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    PrepareGpuAccelerator();
    const int status = LiteRtBenchmarkMain(static_cast<int>(argv.size()), argv.data());
    fflush(stderr);
    // The driver polls for this file, then pulls benchmark.log and results.pb.
    [[NSString stringWithFormat:@"%d\n", status]
        writeToFile:DocumentsPath(@"benchmark.done")
         atomically:YES
           encoding:NSUTF8StringEncoding
              error:nil];
    NSString* log = [NSString stringWithContentsOfFile:DocumentsPath(@"benchmark.log")
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.logText = log ?: @"(no log written)";
      self.textView.text = self.logText;
    });
    if (LaunchedWithFlags()) {
      // Give the console reader a moment to drain, then end the process so the driver returns.
      sleep(1);
      exit(status);
    }
  });
  return YES;
}

@end
