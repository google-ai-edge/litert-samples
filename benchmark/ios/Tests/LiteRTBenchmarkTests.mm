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

#import <XCTest/XCTest.h>

#import "BenchmarkRun.h"

// The benchmark as an XCTest, for a device lab that runs iOS tests (Developer Device Platform's
// `sessions submit xctest`, driven by run_ddp_ios.py). The lab pushes the model and
// Documents/benchmark_args.json, a JSON array of benchmark_model flags such as
//   ["--graph=mobilenet_v2.tflite", "--use_gpu=true"]
// into the app container before the test and pulls Documents/out/ after it. The test runs the tool
// once with those flags, in this process, and writes out/{stdout.txt, results.pb, runtime_info.pb,
// benchmark.done}: the files run_ddp_ios.py lays out as a job directory for the leaderboard
// driver. It fails when the tool exits non-zero. One run per test, so each accelerator gets a
// process of its own, as on the other platforms.
@interface LiteRTBenchmarkTests : XCTestCase
@end

namespace {

NSString* Listing(NSString* dir) {
  NSArray<NSString*>* names =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:dir error:nil];
  return names ? [names componentsJoinedByString:@", "] : @"(unreadable)";
}

}  // namespace

@implementation LiteRTBenchmarkTests

- (void)testBenchmark {
  NSString* argsPath = LiteRtBenchmarkDocumentsPath(@"benchmark_args.json");
  NSData* data = [NSData dataWithContentsOfFile:argsPath];
  if (data == nil) {
    XCTFail(@"no %@; Documents holds [%@], the container %@ holds [%@]", argsPath,
            Listing(LiteRtBenchmarkDocumentsPath(@"")), NSHomeDirectory(),
            Listing(NSHomeDirectory()));
    return;
  }
  NSError* error = nil;
  NSArray* flags = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  bool valid = [flags isKindOfClass:NSArray.class] && flags.count > 0;
  for (id flag in valid ? flags : @[]) {
    valid = valid && [flag isKindOfClass:NSString.class] && [flag hasPrefix:@"--"];
  }
  if (!valid) {
    XCTFail(@"%@ is not a non-empty JSON array of \"--flag=value\" strings: %@", argsPath,
            error.localizedDescription ?: @"");
    return;
  }
  NSFileManager* files = NSFileManager.defaultManager;
  NSString* outDir = LiteRtBenchmarkDocumentsPath(@"out");
  // Nothing from an earlier run may pass as this run's output.
  [files removeItemAtPath:outDir error:nil];
  [files createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:nil];
  NSMutableArray<NSString*>* all = [flags mutableCopy];
  [all addObject:@"--result_file_path=out/results.pb"];
  [all addObject:@"--model_runtime_info_output_file=out/runtime_info.pb"];
  const int status = LiteRtBenchmarkRun(all, @"out/stdout.txt");
  [[NSString stringWithFormat:@"%d\n", status]
      writeToFile:LiteRtBenchmarkDocumentsPath(@"out/benchmark.done")
       atomically:YES
         encoding:NSUTF8StringEncoding
            error:nil];
  XCTAssertEqual(status, 0, @"benchmark_model exited with %d, see out/stdout.txt", status);
}

@end
