// Copyright 2026 Daisuke Majima. All Rights Reserved.
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
// =============================================================================

// LiteRT CompiledModel wrappers shared by the Bonsai macOS and iOS apps.
// BonsaiRuntime owns the LiteRtEnvironment; its RuntimeLibraryDir points at
// the app's Frameworks dir so the registry auto-loads the Metal accelerator
// (no explicit registration call needed on macOS, unlike iOS).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface BonsaiRuntime : NSObject
/// dir must contain libLiteRtMetalAccelerator.dylib (libLiteRt is linked).
- (nullable instancetype)initWithLibraryDir:(NSString *)dir
                                      error:(NSError **)error;
@end

/// One fixed-shape .tflite graph compiled for CPU (XNNPACK) or GPU (Metal,
/// fp32 forced — default fp16 corrupts the Bonsai DiT activations).
@interface BonsaiGraph : NSObject
@property(nonatomic, readonly) double loadSeconds;
@property(nonatomic, readonly) double compileSeconds;
@property(nonatomic, readonly) BOOL fullyAccelerated;
@property(nonatomic, readonly) NSUInteger inputCount;

/// intInputMask: bit k set means host input k (= signature input args_k)
/// is int32; all other inputs and the output are float32.
- (nullable instancetype)initWithRuntime:(BonsaiRuntime *)runtime
                               modelPath:(NSString *)path
                                  useGpu:(BOOL)useGpu
                                 threads:(int)threads
                            intInputMask:(NSUInteger)intInputMask
                                   error:(NSError **)error;

/// inputs[k] feeds signature input args_k; returns output tensor 0 bytes.
- (nullable NSData *)runWithInputs:(NSArray<NSData *> *)inputs
                             error:(NSError **)error
    NS_SWIFT_NAME(run(inputs:));
@end

NS_ASSUME_NONNULL_END
