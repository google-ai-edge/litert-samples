// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
// ==============================================================================

// A LiteRT runtime for the browser that executes on WebGPU through LiteRT.js.
//
// The LiteRT C++ API (litert/cc: Environment, CompiledModel, TensorBuffer,
// ...) calls the runtime through one function table,
// LiteRtRuntimeCApiStruct, obtained from GetLiteRtRuntimeBuiltin(). Natively
// that is the LiteRT runtime library. Here it is this file: model
// introspection (signatures, tensor types) parses the .tflite flatbuffer in
// C++, and the parts that need a GPU — compiling, running a signature,
// allocating / reading / writing buffers — go to LiteRT.js on its WebGPU
// device through a small JS library (litert_js_bridge.js). Tensor buffers are
// WebGPU buffers, so a ModelChain's intermediate tensors stay on the GPU.
//
// Unused table entries fail loudly with their name (runtime_stubs.inc).

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/c/internal/litert_runtime_c_api.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "tflite/schema/mutable/schema_generated.h"

// ---------------------------------------------------------------- JS bridge
extern "C" {
// litert_js_bridge.js. *_async functions suspend the wasm stack (JSPI).
int lrtjs_compile(const uint8_t* bytes, size_t size, int gpu);
void lrtjs_destroy_model(int model);
int lrtjs_fully_accelerated(int model);
int lrtjs_run(int model, const char* signature, const char* input_names,
              const int* input_buffers, int num_inputs,
              const char* output_names, const int* output_buffers,
              const int* output_bytes, int num_outputs);
int lrtjs_create_buffer(size_t bytes);
void lrtjs_destroy_buffer(int buffer);
void lrtjs_write(int buffer, const void* src, size_t bytes);
int lrtjs_read(int buffer, void* dst, size_t bytes);
}

// ---------------------------------------------------------------- objects
class LiteRtEnvironmentT {};
class LiteRtEnvironmentOptionsT {};
class LiteRtOptionsT {
 public:
  LiteRtHwAcceleratorSet accelerators = kLiteRtHwAcceleratorCpu;
};

class LiteRtTensorT {
 public:
  std::string name;
  uint32_t index = 0;  // in its subgraph
  LiteRtRankedTensorType type{};
};

class LiteRtSubgraphT {};

class LiteRtSignatureT {
 public:
  std::string key;
  std::vector<std::string> input_names, output_names;
  std::vector<LiteRtTensorT*> inputs, outputs;
};

class LiteRtModelT {
 public:
  std::vector<uint8_t> bytes;
  std::vector<std::unique_ptr<LiteRtTensorT>> tensors;
  std::vector<std::unique_ptr<LiteRtSignatureT>> signatures;
};

class LiteRtCompiledModelT {
 public:
  LiteRtModelT* model = nullptr;
  int js = 0;
};

class LiteRtTensorBufferRequirementsT {
 public:
  std::vector<LiteRtTensorBufferType> types;
  size_t size = 0;
  std::vector<uint32_t> strides;
  size_t alignment = 64;
};

class LiteRtTensorBufferT {
 public:
  int refs = 1;
  LiteRtTensorBufferType type = kLiteRtTensorBufferTypeHostMemory;
  LiteRtRankedTensorType tensor_type{};
  size_t size = 0;
  int js = 0;             // WebGPU buffer id (0 = host memory)
  void* host = nullptr;   // host memory, or the lock mirror of a GPU buffer
  bool owns_host = false;
  LiteRtHostMemoryDeallocator deallocator = nullptr;
  LiteRtTensorBufferLockMode lock_mode = kLiteRtTensorBufferLockModeRead;
};

namespace {

bool IsWebGpu(LiteRtTensorBufferType t) {
  return t == kLiteRtTensorBufferTypeWebGpuBuffer ||
         t == kLiteRtTensorBufferTypeWebGpuBufferPacked;
}

size_t ElementBytes(LiteRtElementType t) {
  switch (t) {
    case kLiteRtElementTypeFloat16:
    case kLiteRtElementTypeInt16:
      return 2;
    case kLiteRtElementTypeBool:
    case kLiteRtElementTypeInt8:
    case kLiteRtElementTypeUInt8:
      return 1;
    case kLiteRtElementTypeInt64:
      return 8;
    default:
      return 4;
  }
}

size_t TensorBytes(const LiteRtRankedTensorType& t) {
  size_t n = ElementBytes(t.element_type);
  for (unsigned i = 0; i < t.layout.rank; ++i) n *= t.layout.dimensions[i];
  return n;
}

LiteRtElementType FromTflite(tflite::TensorType t) {
  switch (t) {
    case tflite::TensorType_FLOAT32: return kLiteRtElementTypeFloat32;
    case tflite::TensorType_FLOAT16: return kLiteRtElementTypeFloat16;
    case tflite::TensorType_INT32: return kLiteRtElementTypeInt32;
    case tflite::TensorType_INT64: return kLiteRtElementTypeInt64;
    case tflite::TensorType_INT8: return kLiteRtElementTypeInt8;
    case tflite::TensorType_UINT8: return kLiteRtElementTypeUInt8;
    case tflite::TensorType_BOOL: return kLiteRtElementTypeBool;
    case tflite::TensorType_INT16: return kLiteRtElementTypeInt16;
    default: return kLiteRtElementTypeNone;
  }
}

std::string Join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
  return s;
}

LiteRtStatus Fail(const char* what) {
  std::fprintf(stderr, "[litert-js] %s\n", what);
  return kLiteRtStatusErrorRuntimeFailure;
}

// Parses signatures (names + tensor types) from a .tflite flatbuffer.
LiteRtStatus ParseModel(LiteRtModelT& m) {
  const tflite::Model* model = tflite::GetModel(m.bytes.data());
  if (!model || !model->subgraphs()) return Fail("not a .tflite model");
  if (!model->signature_defs()) return Fail("model has no signatures");
  auto make_tensor = [&](const tflite::SubGraph* sg, uint32_t index,
                         const std::string& name) -> LiteRtTensorT* {
    auto t = std::make_unique<LiteRtTensorT>();
    t->name = name;
    t->index = index;
    const tflite::Tensor* ft = sg->tensors()->Get(index);
    t->type.element_type = FromTflite(ft->type());
    const auto* shape = ft->shape();
    t->type.layout.rank = shape ? shape->size() : 0;
    for (unsigned i = 0; i < t->type.layout.rank; ++i) {
      t->type.layout.dimensions[i] = shape->Get(i);
    }
    m.tensors.push_back(std::move(t));
    return m.tensors.back().get();
  };
  for (const tflite::SignatureDef* def : *model->signature_defs()) {
    auto sig = std::make_unique<LiteRtSignatureT>();
    sig->key = def->signature_key() ? def->signature_key()->str() : "";
    const tflite::SubGraph* sg = model->subgraphs()->Get(def->subgraph_index());
    if (def->inputs()) {
      for (const tflite::TensorMap* in : *def->inputs()) {
        sig->input_names.push_back(in->name()->str());
        sig->inputs.push_back(make_tensor(sg, in->tensor_index(), in->name()->str()));
      }
    }
    if (def->outputs()) {
      for (const tflite::TensorMap* out : *def->outputs()) {
        sig->output_names.push_back(out->name()->str());
        sig->outputs.push_back(make_tensor(sg, out->tensor_index(), out->name()->str()));
      }
    }
    m.signatures.push_back(std::move(sig));
  }
  return kLiteRtStatusOk;
}

LiteRtTensorBufferRequirementsT* GpuRequirements(const LiteRtTensorT* t) {
  auto* r = new LiteRtTensorBufferRequirementsT();
  r->types = {kLiteRtTensorBufferTypeWebGpuBuffer};
  r->size = TensorBytes(t->type);
  return r;
}

void FreeBuffer(LiteRtTensorBufferT* b) {
  if (b->js) lrtjs_destroy_buffer(b->js);
  if (b->host) {
    if (b->deallocator) {
      b->deallocator(b->host);
    } else if (b->owns_host) {
      std::free(b->host);
    }
  }
  delete b;
}

// ---------------------------------------------------------------- stubs
template <typename R>
R StubResult(const char* name) {
  std::fprintf(stderr, "[litert-js] unsupported runtime call: %s\n", name);
  if constexpr (std::is_same_v<R, LiteRtStatus>) {
    return kLiteRtStatusErrorUnsupported;
  } else if constexpr (!std::is_void_v<R>) {
    return R{};
  }
}
template <typename F>
struct ReturnOf;
template <typename R, typename... A>
struct ReturnOf<R (*)(A...)> {
  using type = R;
};

LiteRtRuntimeCApiStruct MakeTable() {
  LiteRtRuntimeCApiStruct t{};
#define LITERT_STUB(field)                                             \
  t.field = [](auto...) ->                                             \
      typename ReturnOf<decltype(LiteRtRuntimeCApiStruct::field)>::type { \
        return StubResult<typename ReturnOf<                           \
            decltype(LiteRtRuntimeCApiStruct::field)>::type>(#field);  \
      };
#define LITERT_STUB_VARIADIC(field) t.field = nullptr;
#include "runtime_stubs.inc"
#undef LITERT_STUB
#undef LITERT_STUB_VARIADIC

  // ---- environment
  t.litert_create_environment = [](int, const LiteRtEnvOption*,
                                   LiteRtEnvironment* env) {
    *env = new LiteRtEnvironmentT();
    return kLiteRtStatusOk;
  };
  t.litert_destroy_environment = [](LiteRtEnvironment env) { delete env; };
  t.litert_get_environment_options = [](LiteRtEnvironment,
                                        LiteRtEnvironmentOptions* o) {
    static LiteRtEnvironmentOptionsT options;
    *o = &options;
    return kLiteRtStatusOk;
  };
  t.litert_get_environment_options_value =
      [](LiteRtEnvironmentOptions, LiteRtEnvOptionTag, LiteRtAny*) {
        return kLiteRtStatusErrorNotFound;
      };
  t.litert_environment_supports_fp16 = [](LiteRtEnvironment, bool* s) {
    *s = true;
    return kLiteRtStatusOk;
  };
  t.litert_environment_has_gpu_environment = [](LiteRtEnvironment, bool* h) {
    *h = true;
  };
  for (auto* f : {&t.litert_environment_supports_cl_gl_interop,
                  &t.litert_environment_supports_ahwb_cl_interop,
                  &t.litert_environment_supports_ahwb_gl_interop}) {
    *f = [](LiteRtEnvironment, bool* s) {
      *s = false;
      return kLiteRtStatusOk;
    };
  }
  t.litert_get_num_accelerators = [](LiteRtEnvironment, LiteRtParamIndex* n) {
    *n = 0;
    return kLiteRtStatusOk;
  };

  // ---- options
  t.litert_create_options = [](LiteRtOptions* o) {
    *o = new LiteRtOptionsT();
    return kLiteRtStatusOk;
  };
  t.litert_destroy_options = [](LiteRtOptions o) { delete o; };
  t.litert_set_options_hardware_accelerators =
      [](LiteRtOptions o, LiteRtHwAcceleratorSet a) {
        o->accelerators = a;
        return kLiteRtStatusOk;
      };
  t.litert_get_options_hardware_accelerators =
      [](LiteRtOptions o, LiteRtHwAcceleratorSet* a) {
        *a = o->accelerators;
        return kLiteRtStatusOk;
      };
  t.litert_get_opaque_options = [](LiteRtOptions, LiteRtOpaqueOptions* o) {
    *o = nullptr;
    return kLiteRtStatusOk;
  };

  // ---- model
  t.litert_create_model_from_buffer = [](LiteRtEnvironment, const void* addr,
                                         size_t size, LiteRtModel* model) {
    auto m = std::make_unique<LiteRtModelT>();
    const auto* p = static_cast<const uint8_t*>(addr);
    m->bytes.assign(p, p + size);
    if (auto st = ParseModel(*m); st != kLiteRtStatusOk) return st;
    *model = m.release();
    return kLiteRtStatusOk;
  };
  t.litert_create_model_from_file = [](LiteRtEnvironment, const char* path,
                                       LiteRtModel* model) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return Fail("cannot open model file");
    auto m = std::make_unique<LiteRtModelT>();
    m->bytes.assign(std::istreambuf_iterator<char>(f), {});
    if (auto st = ParseModel(*m); st != kLiteRtStatusOk) return st;
    *model = m.release();
    return kLiteRtStatusOk;
  };
  t.litert_destroy_model = [](LiteRtModel m) { delete m; };
  t.litert_get_num_model_signatures = [](LiteRtModel m, LiteRtParamIndex* n) {
    *n = m->signatures.size();
    return kLiteRtStatusOk;
  };
  t.litert_get_model_signature = [](LiteRtModel m, LiteRtParamIndex i,
                                    LiteRtSignature* s) {
    if (i >= m->signatures.size()) return kLiteRtStatusErrorIndexOOB;
    *s = m->signatures[i].get();
    return kLiteRtStatusOk;
  };
  t.litert_get_num_model_subgraphs = [](LiteRtModel m, LiteRtParamIndex* n) {
    *n = m->signatures.size();
    return kLiteRtStatusOk;
  };
  t.litert_get_main_model_subgraph_index = [](LiteRtModel, LiteRtParamIndex* i) {
    *i = 0;
    return kLiteRtStatusOk;
  };
  t.litert_get_signature_key = [](LiteRtSignature s, const char** key) {
    *key = s->key.c_str();
    return kLiteRtStatusOk;
  };
  t.litert_get_num_signature_inputs = [](LiteRtSignature s, LiteRtParamIndex* n) {
    *n = s->inputs.size();
    return kLiteRtStatusOk;
  };
  t.litert_get_num_signature_outputs = [](LiteRtSignature s, LiteRtParamIndex* n) {
    *n = s->outputs.size();
    return kLiteRtStatusOk;
  };
  t.litert_get_signature_input_name = [](LiteRtSignature s, LiteRtParamIndex i,
                                         const char** name) {
    if (i >= s->input_names.size()) return kLiteRtStatusErrorIndexOOB;
    *name = s->input_names[i].c_str();
    return kLiteRtStatusOk;
  };
  t.litert_get_signature_output_name = [](LiteRtSignature s, LiteRtParamIndex i,
                                          const char** name) {
    if (i >= s->output_names.size()) return kLiteRtStatusErrorIndexOOB;
    *name = s->output_names[i].c_str();
    return kLiteRtStatusOk;
  };
  t.litert_get_signature_input_tensor_by_index =
      [](LiteRtSignature s, LiteRtParamIndex i, LiteRtTensor* tensor) {
        if (i >= s->inputs.size()) return kLiteRtStatusErrorIndexOOB;
        *tensor = s->inputs[i];
        return kLiteRtStatusOk;
      };
  t.litert_get_signature_output_tensor_by_index =
      [](LiteRtSignature s, LiteRtParamIndex i, LiteRtTensor* tensor) {
        if (i >= s->outputs.size()) return kLiteRtStatusErrorIndexOOB;
        *tensor = s->outputs[i];
        return kLiteRtStatusOk;
      };
  t.litert_get_tensor_name = [](LiteRtTensor tensor, const char** name) {
    *name = tensor->name.c_str();
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_index = [](LiteRtTensor tensor, uint32_t* index) {
    *index = tensor->index;
    return kLiteRtStatusOk;
  };
  t.litert_get_quantization_type_id = [](LiteRtTensor,
                                         LiteRtQuantizationTypeId* id) {
    *id = kLiteRtQuantizationNone;  // the Tensor API models are float
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_type_id = [](LiteRtTensor, LiteRtTensorTypeId* id) {
    *id = kLiteRtRankedTensorType;
    return kLiteRtStatusOk;
  };
  t.litert_get_ranked_tensor_type = [](LiteRtTensor tensor,
                                       LiteRtRankedTensorType* type) {
    *type = tensor->type;
    return kLiteRtStatusOk;
  };

  // ---- compiled model
  t.litert_create_compiled_model = [](LiteRtEnvironment, LiteRtModel model,
                                      LiteRtOptions options,
                                      LiteRtCompiledModel* compiled) {
    const bool gpu =
        !options || (options->accelerators & kLiteRtHwAcceleratorGpu) != 0;
    const int js = lrtjs_compile(model->bytes.data(), model->bytes.size(), gpu);
    if (js <= 0) return Fail("LiteRT.js failed to compile the model");
    auto* c = new LiteRtCompiledModelT();
    c->model = model;
    c->js = js;
    *compiled = c;
    // LiteRT.js holds the weights now; signatures were parsed already.
    std::vector<uint8_t>().swap(model->bytes);
    return kLiteRtStatusOk;
  };
  t.litert_destroy_compiled_model = [](LiteRtCompiledModel c) {
    lrtjs_destroy_model(c->js);
    delete c;
  };
  t.litert_compiled_model_is_fully_accelerated = [](LiteRtCompiledModel c,
                                                    bool* full) {
    *full = lrtjs_fully_accelerated(c->js) != 0;
    return kLiteRtStatusOk;
  };
  t.litert_get_compiled_model_input_buffer_requirements =
      [](LiteRtCompiledModel c, LiteRtParamIndex sig, LiteRtParamIndex i,
         LiteRtTensorBufferRequirements* r) {
        const auto& s = *c->model->signatures.at(sig);
        if (i >= s.inputs.size()) return kLiteRtStatusErrorIndexOOB;
        *r = GpuRequirements(s.inputs[i]);
        return kLiteRtStatusOk;
      };
  t.litert_get_compiled_model_output_buffer_requirements =
      [](LiteRtCompiledModel c, LiteRtParamIndex sig, LiteRtParamIndex i,
         LiteRtTensorBufferRequirements* r) {
        const auto& s = *c->model->signatures.at(sig);
        if (i >= s.outputs.size()) return kLiteRtStatusErrorIndexOOB;
        *r = GpuRequirements(s.outputs[i]);
        return kLiteRtStatusOk;
      };
  t.litert_run_compiled_model = [](LiteRtCompiledModel c, LiteRtParamIndex sig,
                                   size_t n_in, LiteRtTensorBuffer* ins,
                                   size_t n_out, LiteRtTensorBuffer* outs) {
    const auto& s = *c->model->signatures.at(sig);
    if (n_in != s.inputs.size() || n_out != s.outputs.size()) {
      return Fail("run: buffer count does not match the signature");
    }
    std::vector<int> in_ids(n_in), out_ids(n_out), out_bytes(n_out);
    for (size_t i = 0; i < n_in; ++i) {
      if (!ins[i]->js) return Fail("run: inputs must be WebGPU buffers");
      in_ids[i] = ins[i]->js;
    }
    for (size_t i = 0; i < n_out; ++i) {
      if (!outs[i]->js) return Fail("run: outputs must be WebGPU buffers");
      out_ids[i] = outs[i]->js;
      out_bytes[i] = static_cast<int>(TensorBytes(s.outputs[i]->type));
    }
    const std::string in_names = Join(s.input_names);
    const std::string out_names = Join(s.output_names);
    const int ok = lrtjs_run(c->js, s.key.c_str(), in_names.c_str(),
                             in_ids.data(), static_cast<int>(n_in),
                             out_names.c_str(), out_ids.data(),
                             out_bytes.data(), static_cast<int>(n_out));
    return ok ? kLiteRtStatusOk : Fail("LiteRT.js run failed");
  };

  // ---- buffer requirements
  t.litert_create_tensor_buffer_requirements_with_alignment =
      [](int n, const LiteRtTensorBufferType* types, size_t size, int n_strides,
         const uint32_t* strides, size_t alignment,
         LiteRtTensorBufferRequirements* r) {
        auto* req = new LiteRtTensorBufferRequirementsT();
        req->types.assign(types, types + n);
        req->size = size;
        if (strides) req->strides.assign(strides, strides + n_strides);
        req->alignment = alignment;
        *r = req;
        return kLiteRtStatusOk;
      };
  t.litert_create_tensor_buffer_requirements =
      [](int n, const LiteRtTensorBufferType* types, size_t size, int n_strides,
         const uint32_t* strides, LiteRtTensorBufferRequirements* r) {
        auto* req = new LiteRtTensorBufferRequirementsT();
        req->types.assign(types, types + n);
        req->size = size;
        if (strides) req->strides.assign(strides, strides + n_strides);
        *r = req;
        return kLiteRtStatusOk;
      };
  t.litert_get_num_tensor_buffer_requirements_supported_buffer_types =
      [](LiteRtTensorBufferRequirements r, int* n) {
        *n = static_cast<int>(r->types.size());
        return kLiteRtStatusOk;
      };
  t.litert_get_tensor_buffer_requirements_supported_tensor_buffer_type =
      [](LiteRtTensorBufferRequirements r, int i, LiteRtTensorBufferType* type) {
        if (i < 0 || i >= static_cast<int>(r->types.size())) {
          return kLiteRtStatusErrorIndexOOB;
        }
        *type = r->types[i];
        return kLiteRtStatusOk;
      };
  t.litert_get_tensor_buffer_requirements_buffer_size =
      [](LiteRtTensorBufferRequirements r, size_t* size) {
        *size = r->size;
        return kLiteRtStatusOk;
      };
  t.litert_get_tensor_buffer_requirements_strides =
      [](LiteRtTensorBufferRequirements r, int* n, const uint32_t** strides) {
        *n = static_cast<int>(r->strides.size());
        *strides = r->strides.data();
        return kLiteRtStatusOk;
      };
  t.litert_get_tensor_buffer_requirements_alignment =
      [](LiteRtTensorBufferRequirements r, size_t* alignment) {
        *alignment = r->alignment;
        return kLiteRtStatusOk;
      };
  t.litert_destroy_tensor_buffer_requirements =
      [](LiteRtTensorBufferRequirements r) { delete r; };

  // ---- tensor buffers
  t.litert_create_managed_tensor_buffer =
      [](LiteRtEnvironment, LiteRtTensorBufferType type,
         const LiteRtRankedTensorType* tensor_type, size_t size,
         LiteRtTensorBuffer* out) {
        auto* b = new LiteRtTensorBufferT();
        b->tensor_type = *tensor_type;
        b->size = size ? size : TensorBytes(*tensor_type);
        if (IsWebGpu(type)) {
          b->type = kLiteRtTensorBufferTypeWebGpuBuffer;
          b->js = lrtjs_create_buffer(b->size);
          if (b->js <= 0) {
            delete b;
            return Fail("WebGPU buffer allocation failed");
          }
        } else if (type == kLiteRtTensorBufferTypeHostMemory) {
          b->type = type;
          b->host = std::calloc(1, b->size);
          b->owns_host = true;
        } else {
          delete b;
          return Fail("unsupported tensor buffer type");
        }
        *out = b;
        return kLiteRtStatusOk;
      };
  t.litert_create_managed_tensor_buffer_from_requirements =
      [](LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
         LiteRtTensorBufferRequirements r, LiteRtTensorBuffer* out) {
        const LiteRtTensorBufferType type =
            r->types.empty() ? kLiteRtTensorBufferTypeWebGpuBuffer : r->types[0];
        return GetLiteRtRuntimeBuiltin()->litert_create_managed_tensor_buffer(
            env, type, tensor_type, r->size, out);
      };
  t.litert_create_tensor_buffer_from_host_memory =
      [](const LiteRtRankedTensorType* tensor_type, void* addr, size_t size,
         LiteRtHostMemoryDeallocator deallocator, LiteRtTensorBuffer* out) {
        auto* b = new LiteRtTensorBufferT();
        b->type = kLiteRtTensorBufferTypeHostMemory;
        b->tensor_type = *tensor_type;
        b->size = size;
        b->host = addr;
        b->deallocator = deallocator;
        *out = b;
        return kLiteRtStatusOk;
      };
  t.litert_duplicate_tensor_buffer = [](LiteRtTensorBuffer b) {
    ++b->refs;
    return kLiteRtStatusOk;
  };
  t.litert_destroy_tensor_buffer = [](LiteRtTensorBuffer b) {
    if (--b->refs == 0) FreeBuffer(b);
  };
  t.litert_get_tensor_buffer_type = [](LiteRtTensorBuffer b,
                                       LiteRtTensorBufferType* type) {
    *type = b->type;
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_tensor_type = [](LiteRtTensorBuffer b,
                                              LiteRtRankedTensorType* type) {
    *type = b->tensor_type;
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_size = [](LiteRtTensorBuffer b, size_t* size) {
    *size = b->size;
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_packed_size = [](LiteRtTensorBuffer b,
                                              size_t* size) {
    *size = TensorBytes(b->tensor_type);
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_offset = [](LiteRtTensorBuffer, size_t* offset) {
    *offset = 0;
    return kLiteRtStatusOk;
  };
  t.litert_has_tensor_buffer_event = [](LiteRtTensorBuffer, bool* has) {
    *has = false;
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_host_memory = [](LiteRtTensorBuffer b,
                                              void** addr) {
    if (b->js) return Fail("host memory of a WebGPU buffer: lock it");
    *addr = b->host;
    return kLiteRtStatusOk;
  };
  t.litert_get_tensor_buffer_web_gpu_buffer = [](LiteRtTensorBuffer b,
                                                 HwMemoryHandle* handle) {
    if (!b->js) return kLiteRtStatusErrorInvalidArgument;
    *handle = reinterpret_cast<HwMemoryHandle>(static_cast<intptr_t>(b->js));
    return kLiteRtStatusOk;
  };
  t.litert_lock_tensor_buffer = [](LiteRtTensorBuffer b, void** addr,
                                   LiteRtTensorBufferLockMode mode) {
    if (b->js) {
      if (!b->host) {
        b->host = std::malloc(b->size);
        b->owns_host = true;
      }
      if (mode != kLiteRtTensorBufferLockModeWrite &&
          !lrtjs_read(b->js, b->host, TensorBytes(b->tensor_type))) {
        return Fail("WebGPU readback failed");
      }
    }
    b->lock_mode = mode;
    *addr = b->host;
    return kLiteRtStatusOk;
  };
  t.litert_unlock_tensor_buffer = [](LiteRtTensorBuffer b) {
    if (b->js && b->lock_mode != kLiteRtTensorBufferLockModeRead) {
      lrtjs_write(b->js, b->host, TensorBytes(b->tensor_type));
    }
    return kLiteRtStatusOk;
  };

  // ---- layouts
  t.litert_get_num_layout_elements = [](const LiteRtLayout* l, size_t* n) {
    size_t e = 1;
    for (unsigned i = 0; i < l->rank; ++i) e *= l->dimensions[i];
    *n = e;
    return kLiteRtStatusOk;
  };
  t.litert_is_same_layout = [](const LiteRtLayout* a, const LiteRtLayout* b,
                               bool* same) {
    *same = a->rank == b->rank &&
            std::memcmp(a->dimensions, b->dimensions,
                        a->rank * sizeof(int32_t)) == 0;
    return kLiteRtStatusOk;
  };
  return t;
}

}  // namespace

const LiteRtRuntimeCApiStruct* GetLiteRtRuntimeBuiltin() {
  static const LiteRtRuntimeCApiStruct table = MakeTable();
  return &table;
}

// For the embind layer: the WebGPU buffer id behind a tensor buffer.
int LiteRtJsBufferId(LiteRtTensorBuffer b) { return b ? b->js : 0; }
