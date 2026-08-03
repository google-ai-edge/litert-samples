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

// CompiledModel C API call sequence: env(RuntimeLibraryDir) -> model ->
// options(+hand-built TOML opaque options) -> compiled model -> managed
// buffers -> run. On macOS the runtime pair is ai-edge-litert 2.1.6
// (libLiteRt + libLiteRtMetalAccelerator from the same wheel — the pair must
// stay same-generation: cross-generation pairs reject each other's
// serialized options).

#import "LiteRtGraph.h"

#include <cstring>
#include <string>
#include <vector>

// macOS links the raw dylib against downloaded headers ("litert/c/..." path
// style); iOS links the bazel-built CLiteRT.xcframework, which bundles the
// same headers flat inside the framework (<CLiteRT/...> style).
#if __has_include(<CLiteRT/litert_common.h>)
#include <CLiteRT/litert_common.h>
#include <CLiteRT/litert_any.h>
#include <CLiteRT/litert_environment.h>
#include <CLiteRT/litert_environment_options.h>
#include <CLiteRT/litert_model.h>
#include <CLiteRT/litert_model_types.h>
#include <CLiteRT/litert_options.h>
#include <CLiteRT/litert_opaque_options.h>
#include <CLiteRT/litert_compiled_model.h>
#include <CLiteRT/litert_tensor_buffer.h>
#include <CLiteRT/litert_tensor_buffer_requirements.h>
#include <CLiteRT/litert_tensor_buffer_types.h>
#else
#include "litert/c/litert_common.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#endif

static NSError *MakeError(NSString *what, LiteRtStatus status) {
  return [NSError errorWithDomain:@"Bonsai"
                             code:status
                         userInfo:@{
                           NSLocalizedDescriptionKey : [NSString
                               stringWithFormat:@"%@ (status %d)", what, status]
                         }];
}

#define BAIL_IF(expr, what)                          \
  do {                                               \
    LiteRtStatus s_ = (expr);                        \
    if (s_ != kLiteRtStatusOk) {                     \
      if (error) *error = MakeError(what, s_);       \
      return nil;                                    \
    }                                                \
  } while (0)

@implementation BonsaiRuntime {
 @public
  LiteRtEnvironment _env;
  std::string _libDir;  // must outlive the env option struct
}

- (nullable instancetype)initWithLibraryDir:(NSString *)dir
                                      error:(NSError **)error {
  self = [super init];
  if (!self) return nil;
  _libDir = std::string([dir UTF8String]);
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = _libDir.c_str();
  _env = nullptr;
  BAIL_IF(LiteRtCreateEnvironment(1, &opt, &_env), @"CreateEnvironment");
  return self;
}

- (void)dealloc {
  if (_env) LiteRtDestroyEnvironment(_env);
}
@end

@implementation BonsaiGraph {
  BonsaiRuntime *_runtime;  // keeps the environment alive
  LiteRtModel _model;
  LiteRtOptions _options;
  LiteRtCompiledModel _compiled;
  std::vector<LiteRtTensorBuffer> _in;
  LiteRtTensorBuffer _out;
  std::vector<int> _argpos;        // signature position -> host input index
  std::vector<size_t> _inBytes;    // per signature position
  size_t _outBytes;
}

static size_t LayoutBytes(const LiteRtLayout &layout, size_t elemSize) {
  size_t n = 1;
  for (uint32_t d = 0; d < layout.rank; ++d) n *= (size_t)layout.dimensions[d];
  return n * elemSize;
}

- (nullable instancetype)initWithRuntime:(BonsaiRuntime *)runtime
                               modelPath:(NSString *)path
                                  useGpu:(BOOL)useGpu
                                 threads:(int)threads
                            intInputMask:(NSUInteger)intInputMask
                                   error:(NSError **)error {
  self = [super init];
  if (!self) return nil;
  _runtime = runtime;
  LiteRtEnvironment env = runtime->_env;

  NSDate *t0 = [NSDate date];
  BAIL_IF(LiteRtCreateModelFromFile(env, [path UTF8String], &_model),
          @"CreateModelFromFile");
  _loadSeconds = -[t0 timeIntervalSinceNow];

  BAIL_IF(LiteRtCreateOptions(&_options), @"CreateOptions");
  BAIL_IF(LiteRtSetOptionsHardwareAccelerators(
              _options, useGpu ? (kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu)
                               : kLiteRtHwAcceleratorCpu),
          @"SetHardwareAccelerators");
  // The prebuilt runtime does not export the Lrt*Options helpers; the
  // accelerators parse the opaque payload as TOML (precision 2 = fp32).
  const char *ident = useGpu ? "gpu_options" : "xnnpack";
  std::string toml = useGpu ? "precision = 2\n"
                            : "num_threads = " + std::to_string(threads) + "\n";
  LiteRtOpaqueOptions opaque = nullptr;
  BAIL_IF(LiteRtCreateOpaqueOptions(ident, strdup(toml.c_str()), free, &opaque),
          @"CreateOpaqueOptions");
  BAIL_IF(LiteRtAddOpaqueOptions(_options, opaque), @"AddOpaqueOptions");

  t0 = [NSDate date];
  BAIL_IF(LiteRtCreateCompiledModel(env, _model, _options, &_compiled),
          @"CreateCompiledModel");
  _compileSeconds = -[t0 timeIntervalSinceNow];
  bool full = false;
  LiteRtCompiledModelIsFullyAccelerated(_compiled, &full);
  _fullyAccelerated = full;

  LiteRtSignature sig = nullptr;
  BAIL_IF(LiteRtGetModelSignature(_model, 0, &sig), @"GetModelSignature");
  LiteRtParamIndex nIn = 0;
  BAIL_IF(LiteRtGetNumSignatureInputs(sig, &nIn), @"GetNumSignatureInputs");
  _inputCount = nIn;
  for (LiteRtParamIndex i = 0; i < nIn; ++i) {
    const char *name = nullptr;
    BAIL_IF(LiteRtGetSignatureInputName(sig, i, &name), @"GetSignatureInputName");
    const char *p = name ? strstr(name, "args_") : nullptr;
    _argpos.push_back(p ? atoi(p + 5) : (int)i);
  }

  _in.resize(nIn, nullptr);
  _inBytes.resize(nIn, 0);
  for (LiteRtParamIndex i = 0; i < nIn; ++i) {
    LiteRtLayout layout;
    BAIL_IF(LiteRtGetCompiledModelInputTensorLayout(_compiled, 0, i, &layout),
            @"GetInputTensorLayout");
    bool isInt = (intInputMask >> _argpos[i]) & 1;
    _inBytes[i] = LayoutBytes(layout, 4);  // int32 and float32 are both 4 B
    LiteRtRankedTensorType type;
    type.element_type =
        isInt ? kLiteRtElementTypeInt32 : kLiteRtElementTypeFloat32;
    type.layout = layout;
    LiteRtTensorBufferRequirements req = nullptr;
    BAIL_IF(LiteRtGetCompiledModelInputBufferRequirements(_compiled, 0, i, &req),
            @"GetInputBufferRequirements");
    BAIL_IF(LiteRtCreateManagedTensorBufferFromRequirements(env, &type, req,
                                                            &_in[i]),
            @"CreateInputBuffer");
  }
  LiteRtLayout layout;
  BAIL_IF(LiteRtGetCompiledModelOutputTensorLayouts(_compiled, 0, 1, &layout,
                                                    false),
          @"GetOutputTensorLayouts");
  _outBytes = LayoutBytes(layout, 4);
  LiteRtRankedTensorType type;
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout = layout;
  LiteRtTensorBufferRequirements req = nullptr;
  BAIL_IF(LiteRtGetCompiledModelOutputBufferRequirements(_compiled, 0, 0, &req),
          @"GetOutputBufferRequirements");
  BAIL_IF(LiteRtCreateManagedTensorBufferFromRequirements(env, &type, req, &_out),
          @"CreateOutputBuffer");
  return self;
}

- (nullable NSData *)runWithInputs:(NSArray<NSData *> *)inputs
                             error:(NSError **)error {
  if (inputs.count != _in.size()) {
    if (error)
      *error = MakeError([NSString stringWithFormat:@"input count %lu != %zu",
                                                    inputs.count, _in.size()],
                         kLiteRtStatusErrorInvalidArgument);
    return nil;
  }
  for (size_t i = 0; i < _in.size(); ++i) {
    NSData *src = inputs[_argpos[i]];
    if (src.length != _inBytes[i]) {
      if (error)
        *error = MakeError(
            [NSString stringWithFormat:@"input %d byte size %lu != graph %zu",
                                       _argpos[i], src.length, _inBytes[i]],
            kLiteRtStatusErrorInvalidArgument);
      return nil;
    }
    void *host = nullptr;
    BAIL_IF(LiteRtLockTensorBuffer(_in[i], &host,
                                   kLiteRtTensorBufferLockModeWrite),
            @"LockInputBuffer");
    memcpy(host, src.bytes, src.length);
    BAIL_IF(LiteRtUnlockTensorBuffer(_in[i]), @"UnlockInputBuffer");
  }
  BAIL_IF(LiteRtRunCompiledModel(_compiled, 0, _in.size(), _in.data(), 1, &_out),
          @"RunCompiledModel");
  void *host = nullptr;
  BAIL_IF(LiteRtLockTensorBuffer(_out, &host, kLiteRtTensorBufferLockModeRead),
          @"LockOutputBuffer");
  NSData *out = [NSData dataWithBytes:host length:_outBytes];
  BAIL_IF(LiteRtUnlockTensorBuffer(_out), @"UnlockOutputBuffer");
  return out;
}

- (void)dealloc {
  for (auto b : _in)
    if (b) LiteRtDestroyTensorBuffer(b);
  if (_out) LiteRtDestroyTensorBuffer(_out);
  if (_compiled) LiteRtDestroyCompiledModel(_compiled);
  if (_options) LiteRtDestroyOptions(_options);
  if (_model) LiteRtDestroyModel(_model);
}
@end
