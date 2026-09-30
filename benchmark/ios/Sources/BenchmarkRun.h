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

#ifndef LITERT_SAMPLES_BENCHMARK_IOS_BENCHMARK_RUN_H_
#define LITERT_SAMPLES_BENCHMARK_IOS_BENCHMARK_RUN_H_

#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

// Exported from the app binary: Tests/LiteRTBenchmarkTests.mm, loaded into the app, calls these.
#define LITERT_BENCHMARK_EXPORT __attribute__((visibility("default")))

// The app's Documents folder joined with `name`: the folder the driver scripts copy files
// into and out of.
LITERT_BENCHMARK_EXPORT NSString* LiteRtBenchmarkDocumentsPath(NSString* name);

// Runs LiteRT's benchmark_model once, in this process, with `flags` (its command-line flags,
// without argv[0]). Relative values of --graph, --result_file_path,
// --model_runtime_info_output_file and --profiling_output_csv_file resolve against Documents.
// The tool logs to stderr; for the duration of the run that is mirrored into
// Documents/<logName> (whose directory must exist) and echoed on stdout. The embedded Metal
// accelerator is loaded before the first run so --use_gpu=true finds it. Returns the tool's
// exit status.
LITERT_BENCHMARK_EXPORT int LiteRtBenchmarkRun(NSArray<NSString*>* flags, NSString* logName);

#ifdef __cplusplus
}
#endif

#endif  // LITERT_SAMPLES_BENCHMARK_IOS_BENCHMARK_RUN_H_
