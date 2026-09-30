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

#import "BenchmarkRun.h"

#include <unistd.h>

namespace {

// benchmark_model's flags: the app's own launch arguments (anything starting with "--"), or,
// when there are none, the keys of Documents/benchmark_params.json as --key=value. Empty when
// there is neither; the app then runs nothing, since the tool has no model to load.
NSArray<NSString*>* BenchmarkFlags() {
  NSMutableArray<NSString*>* flags = [NSMutableArray array];
  NSArray<NSString*>* launch_args = NSProcessInfo.processInfo.arguments;
  for (NSUInteger i = 1; i < launch_args.count; ++i) {
    if ([launch_args[i] hasPrefix:@"--"]) {
      [flags addObject:launch_args[i]];
    }
  }
  if (flags.count == 0) {
    NSData* data =
        [NSData dataWithContentsOfFile:LiteRtBenchmarkDocumentsPath(@"benchmark_params.json")];
    NSDictionary* params =
        data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    for (NSString* key in params) {
      [flags addObject:[NSString stringWithFormat:@"--%@=%@", key, [params[key] description]]];
    }
  }
  return flags;
}

// Launched from the command line (devicectl or an Xcode scheme) means a driver is waiting for the
// process to end; launched by tapping the icon means a person is reading the screen.
bool LaunchedWithFlags() {
  NSArray<NSString*>* launch_args = NSProcessInfo.processInfo.arguments;
  for (NSUInteger i = 1; i < launch_args.count; ++i) {
    if ([launch_args[i] hasPrefix:@"--"]) {
      return true;
    }
  }
  return false;
}

}  // namespace

@implementation AppDelegate

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  application.idleTimerDisabled = YES;
  NSFileManager* files = [NSFileManager defaultManager];
  if ([files fileExistsAtPath:LiteRtBenchmarkDocumentsPath(@"benchmark_args.json")]) {
    // A lab pushed the flags for Tests/LiteRTBenchmarkTests.mm, which runs the tool itself.
    self.logText = @"benchmark_args.json is for the XCTest; nothing runs at launch.";
    return YES;
  }
  NSArray<NSString*>* flags = BenchmarkFlags();
  if (flags.count == 0) {
    self.logText = @"No benchmark_model flags. Launch the app with them (run_ios.sh) or put a "
                   @"benchmark_params.json in Documents.";
    return YES;
  }
  [files removeItemAtPath:LiteRtBenchmarkDocumentsPath(@"benchmark.done") error:nil];
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    const int status = LiteRtBenchmarkRun(flags, @"benchmark.log");
    // The driver polls for this file, then pulls benchmark.log and results.pb.
    [[NSString stringWithFormat:@"%d\n", status]
        writeToFile:LiteRtBenchmarkDocumentsPath(@"benchmark.done")
         atomically:YES
           encoding:NSUTF8StringEncoding
              error:nil];
    NSString* logPath = LiteRtBenchmarkDocumentsPath(@"benchmark.log");
    NSString* log = [NSString stringWithContentsOfFile:logPath
                                              encoding:NSUTF8StringEncoding
                                                 error:nil];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.logText = log ?: @"(no log written)";
      self.textView.text = self.logText;
    });
    if (LaunchedWithFlags()) {
      // End the process so the driver returns; the log reader has already drained. _exit skips
      // the static destructors, which this background thread must not run under the main one.
      fflush(stdout);
      fflush(stderr);
      _exit(status);
    }
  });
  return YES;
}

@end
