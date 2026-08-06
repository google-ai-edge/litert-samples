/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Temporary JNI bindings over the LiteRT C API's dynamic-shape entry points,
// which the Kotlin API does not expose yet:
//
//   - LiteRtCompiledModelResizeInputTensor   (litert/cc ResizeInputTensor)
//   - LiteRtGetCompiledModelOutputTensorLayouts(update_allocation=true),
//     the documented companion that re-allocates and propagates the resized
//     input shapes to statically-inferable outputs, and
//   - a self-contained dynamic run (resize + managed buffers +
//     LiteRtRunCompiledModel + read-back), needed when an output is a TFLite
//     *dynamic* tensor whose shape only materializes during invoke — for
//     those, buffers created from pre-invoke buffer requirements are stale,
//     so the caller supplies the known output shape instead.
//
// Every symbol used here is exported by the libLiteRt.so that ships inside
// the com.google.ai.edge.litert:litert AAR, so this library only needs dlsym
// — no LiteRT headers or link-time dependency. Once the official Kotlin
// resize API lands, this file and LiteRtDynamicShape.kt can be deleted.

#include <dlfcn.h>
#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Minimal mirror of the LiteRT C API types (litert/c, v2.1.5). LiteRtLayout
// and LiteRtRankedTensorType are documented ABI-stable and static_asserted
// upstream; LiteRtParamIndex is size_t (litert_common.h).
using LiteRtStatus = int;  // kLiteRtStatusOk == 0.
using LiteRtCompiledModel = void*;
using LiteRtTensorBuffer = void*;
using LiteRtEnvironment = void*;
using LiteRtParamIndex = size_t;

constexpr int kMaxRank = 8;

struct LiteRtLayout {
  unsigned int rank : 7;
  bool has_strides : 1;
  int32_t dimensions[kMaxRank];
  uint32_t strides[kMaxRank];
};
static_assert(sizeof(LiteRtLayout) == 68, "LiteRtLayout ABI mismatch");

// litert/c/litert_model_types.h.
struct LiteRtRankedTensorType {
  int32_t element_type;
  LiteRtLayout layout;
};
static_assert(sizeof(LiteRtRankedTensorType) == 72,
              "LiteRtRankedTensorType ABI mismatch");

constexpr int32_t kElementTypeFloat32 = 1;  // kLiteRtElementTypeFloat32.
constexpr int kBufferTypeHostMemory = 1;  // kLiteRtTensorBufferTypeHostMemory.
constexpr int kLockModeRead = 0;          // kLiteRtTensorBufferLockModeRead.
constexpr int kLockModeWrite = 1;         // kLiteRtTensorBufferLockModeWrite.

using ResizeInputTensorFn = LiteRtStatus (*)(LiteRtCompiledModel,
                                             LiteRtParamIndex,
                                             LiteRtParamIndex, const int*,
                                             size_t);
using GetOutputTensorLayoutsFn = LiteRtStatus (*)(LiteRtCompiledModel,
                                                  LiteRtParamIndex, size_t,
                                                  LiteRtLayout*, bool);
using GetEnvironmentFn = LiteRtStatus (*)(LiteRtCompiledModel,
                                          LiteRtEnvironment*);
using CreateManagedTensorBufferFn =
    LiteRtStatus (*)(LiteRtEnvironment, int, const LiteRtRankedTensorType*,
                     size_t, LiteRtTensorBuffer*);
using LockTensorBufferFn = LiteRtStatus (*)(LiteRtTensorBuffer, void**, int);
using UnlockTensorBufferFn = LiteRtStatus (*)(LiteRtTensorBuffer);
using DestroyTensorBufferFn = void (*)(LiteRtTensorBuffer);
using RunCompiledModelFn = LiteRtStatus (*)(LiteRtCompiledModel,
                                            LiteRtParamIndex, size_t,
                                            LiteRtTensorBuffer*, size_t,
                                            LiteRtTensorBuffer*);

ResizeInputTensorFn resize_input_tensor = nullptr;
GetOutputTensorLayoutsFn get_output_tensor_layouts = nullptr;
GetEnvironmentFn get_environment = nullptr;
CreateManagedTensorBufferFn create_managed_tensor_buffer = nullptr;
LockTensorBufferFn lock_tensor_buffer = nullptr;
UnlockTensorBufferFn unlock_tensor_buffer = nullptr;
DestroyTensorBufferFn destroy_tensor_buffer = nullptr;
RunCompiledModelFn run_compiled_model = nullptr;

// CompiledModel's native handle lives in the private `handle` field of its
// Kotlin base class com.google.ai.edge.litert.JniHandle.
jfieldID jni_handle_field = nullptr;

void ThrowRuntimeException(JNIEnv* env, const char* function,
                           LiteRtStatus status) {
  char message[160];
  std::snprintf(message, sizeof(message), "%s failed with LiteRtStatus %d",
                function, status);
  jclass exception_class = env->FindClass("java/lang/RuntimeException");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message);
  }
}

// The Kotlin `handle` field holds a litert::jni::CompiledModelWrapper*
// (litert/kotlin/src/main/jni/litert_model_wrapper.h), not the C handle. The
// wrapper's first member is litert::CompiledModel, whose first base
// BaseHandle<LiteRtCompiledModel> stores the raw C handle as its first word
// (libc++ unique_ptr keeps the pointer first), so one dereference yields the
// LiteRtCompiledModel. Layout pinned to the litert 2.1.5 AAR this sample
// builds against — verified on device; the official Kotlin resize API will
// replace all of this.
LiteRtCompiledModel GetModel(JNIEnv* env, jobject compiled_model) {
  const jlong wrapper = env->GetLongField(compiled_model, jni_handle_field);
  if (wrapper == 0) {
    return nullptr;
  }
  return *reinterpret_cast<LiteRtCompiledModel*>(wrapper);
}

// Reads one IntArray shape into `layout` (dims + rank). Returns false and
// throws if the rank is out of range.
bool ReadShape(JNIEnv* env, jintArray dims, LiteRtLayout* layout) {
  const jsize rank = env->GetArrayLength(dims);
  if (rank > kMaxRank) {
    ThrowRuntimeException(env, "ReadShape(rank)", -1);
    return false;
  }
  layout->rank = static_cast<unsigned int>(rank);
  env->GetIntArrayRegion(dims, 0, rank, layout->dimensions);
  return true;
}

size_t NumElements(const LiteRtLayout& layout) {
  size_t count = 1;
  for (unsigned int i = 0; i < layout.rank; ++i) {
    count *= static_cast<size_t>(layout.dimensions[i]);
  }
  return count;
}

void ResizeInputTensor(JNIEnv* env, jclass /*clazz*/, jobject compiled_model,
                       jint signature_index, jint input_index,
                       jintArray dims) {
  LiteRtLayout layout = {};
  if (!ReadShape(env, dims, &layout)) {
    return;
  }
  const LiteRtStatus status = resize_input_tensor(
      GetModel(env, compiled_model),
      static_cast<LiteRtParamIndex>(signature_index),
      static_cast<LiteRtParamIndex>(input_index), layout.dimensions,
      static_cast<size_t>(layout.rank));
  if (status != 0) {
    ThrowRuntimeException(env, "LiteRtCompiledModelResizeInputTensor", status);
  }
}

jobjectArray UpdateOutputLayouts(JNIEnv* env, jclass /*clazz*/,
                                 jobject compiled_model, jint signature_index,
                                 jint output_count) {
  LiteRtLayout layouts[16] = {};
  if (output_count < 0 || output_count > 16) {
    ThrowRuntimeException(env, "UpdateOutputLayouts(output_count)", -1);
    return nullptr;
  }
  const LiteRtStatus status = get_output_tensor_layouts(
      GetModel(env, compiled_model),
      static_cast<LiteRtParamIndex>(signature_index),
      static_cast<size_t>(output_count), layouts, /*update_allocation=*/true);
  if (status != 0) {
    ThrowRuntimeException(env, "LiteRtGetCompiledModelOutputTensorLayouts",
                          status);
    return nullptr;
  }
  jobjectArray shapes =
      env->NewObjectArray(output_count, env->FindClass("[I"), nullptr);
  for (jint i = 0; i < output_count; ++i) {
    const jsize rank = static_cast<jsize>(layouts[i].rank);
    jintArray shape = env->NewIntArray(rank);
    env->SetIntArrayRegion(shape, 0, rank, layouts[i].dimensions);
    env->SetObjectArrayElement(shapes, i, shape);
    env->DeleteLocalRef(shape);
  }
  return shapes;
}

// One complete dynamic-shape inference: strict-resize every input, create
// float32 host-memory buffers of the given shapes on both sides, run, read
// the outputs back. All buffers are destroyed before returning.
jobjectArray RunDynamic(JNIEnv* env, jclass /*clazz*/, jobject compiled_model,
                        jint signature_index, jobjectArray input_data,
                        jobjectArray input_shapes, jobjectArray output_shapes) {
  LiteRtCompiledModel model = GetModel(env, compiled_model);
  const jsize num_inputs = env->GetArrayLength(input_data);
  const jsize num_outputs = env->GetArrayLength(output_shapes);

  std::vector<LiteRtTensorBuffer> buffers;
  auto destroy_all = [&buffers]() {
    for (LiteRtTensorBuffer buffer : buffers) {
      destroy_tensor_buffer(buffer);
    }
  };

  LiteRtEnvironment litert_env = nullptr;
  LiteRtStatus status = get_environment(model, &litert_env);
  if (status != 0) {
    ThrowRuntimeException(env, "LiteRtGetCompiledModelEnvironment", status);
    return nullptr;
  }

  // Inputs: resize, then create + fill a managed host buffer per input.
  for (jsize i = 0; i < num_inputs; ++i) {
    LiteRtRankedTensorType tensor_type = {};
    tensor_type.element_type = kElementTypeFloat32;
    jintArray dims = static_cast<jintArray>(
        env->GetObjectArrayElement(input_shapes, i));
    if (!ReadShape(env, dims, &tensor_type.layout)) {
      destroy_all();
      return nullptr;
    }
    status = resize_input_tensor(
        model, static_cast<LiteRtParamIndex>(signature_index),
        static_cast<LiteRtParamIndex>(i), tensor_type.layout.dimensions,
        static_cast<size_t>(tensor_type.layout.rank));
    if (status != 0) {
      destroy_all();
      ThrowRuntimeException(env, "LiteRtCompiledModelResizeInputTensor",
                            status);
      return nullptr;
    }

    const size_t bytes = NumElements(tensor_type.layout) * sizeof(float);
    LiteRtTensorBuffer buffer = nullptr;
    status = create_managed_tensor_buffer(litert_env, kBufferTypeHostMemory,
                                          &tensor_type, bytes, &buffer);
    if (status != 0) {
      destroy_all();
      ThrowRuntimeException(env, "LiteRtCreateManagedTensorBuffer", status);
      return nullptr;
    }
    buffers.push_back(buffer);

    jfloatArray data = static_cast<jfloatArray>(
        env->GetObjectArrayElement(input_data, i));
    void* host_memory = nullptr;
    status = lock_tensor_buffer(buffer, &host_memory, kLockModeWrite);
    if (status != 0) {
      destroy_all();
      ThrowRuntimeException(env, "LiteRtLockTensorBuffer", status);
      return nullptr;
    }
    env->GetFloatArrayRegion(data, 0, static_cast<jsize>(bytes / sizeof(float)),
                             static_cast<jfloat*>(host_memory));
    unlock_tensor_buffer(buffer);
  }

  // Outputs: a managed host buffer of the caller-provided (known) shape.
  std::vector<size_t> output_elements(num_outputs);
  for (jsize i = 0; i < num_outputs; ++i) {
    LiteRtRankedTensorType tensor_type = {};
    tensor_type.element_type = kElementTypeFloat32;
    jintArray dims = static_cast<jintArray>(
        env->GetObjectArrayElement(output_shapes, i));
    if (!ReadShape(env, dims, &tensor_type.layout)) {
      destroy_all();
      return nullptr;
    }
    output_elements[i] = NumElements(tensor_type.layout);
    LiteRtTensorBuffer buffer = nullptr;
    status = create_managed_tensor_buffer(
        litert_env, kBufferTypeHostMemory, &tensor_type,
        output_elements[i] * sizeof(float), &buffer);
    if (status != 0) {
      destroy_all();
      ThrowRuntimeException(env, "LiteRtCreateManagedTensorBuffer", status);
      return nullptr;
    }
    buffers.push_back(buffer);
  }

  status = run_compiled_model(
      model, static_cast<LiteRtParamIndex>(signature_index),
      static_cast<size_t>(num_inputs), buffers.data(),
      static_cast<size_t>(num_outputs), buffers.data() + num_inputs);
  if (status != 0) {
    destroy_all();
    ThrowRuntimeException(env, "LiteRtRunCompiledModel", status);
    return nullptr;
  }

  jobjectArray results =
      env->NewObjectArray(num_outputs, env->FindClass("[F"), nullptr);
  for (jsize i = 0; i < num_outputs; ++i) {
    LiteRtTensorBuffer buffer = buffers[num_inputs + i];
    void* host_memory = nullptr;
    status = lock_tensor_buffer(buffer, &host_memory, kLockModeRead);
    if (status != 0) {
      destroy_all();
      ThrowRuntimeException(env, "LiteRtLockTensorBuffer", status);
      return nullptr;
    }
    jfloatArray data = env->NewFloatArray(
        static_cast<jsize>(output_elements[i]));
    env->SetFloatArrayRegion(data, 0, static_cast<jsize>(output_elements[i]),
                             static_cast<const jfloat*>(host_memory));
    unlock_tensor_buffer(buffer);
    env->SetObjectArrayElement(results, i, data);
    env->DeleteLocalRef(data);
  }
  destroy_all();
  return results;
}

bool Resolve(JNIEnv* env) {
  // libLiteRt.so ships in the same APK (from the litert AAR) and is usually
  // already loaded by the Kotlin API; dlopen either way is idempotent.
  void* lib = dlopen("libLiteRt.so", RTLD_NOW);
  if (lib == nullptr) {
    return false;
  }
  resize_input_tensor = reinterpret_cast<ResizeInputTensorFn>(
      dlsym(lib, "LiteRtCompiledModelResizeInputTensor"));
  get_output_tensor_layouts = reinterpret_cast<GetOutputTensorLayoutsFn>(
      dlsym(lib, "LiteRtGetCompiledModelOutputTensorLayouts"));
  get_environment = reinterpret_cast<GetEnvironmentFn>(
      dlsym(lib, "LiteRtGetCompiledModelEnvironment"));
  create_managed_tensor_buffer = reinterpret_cast<CreateManagedTensorBufferFn>(
      dlsym(lib, "LiteRtCreateManagedTensorBuffer"));
  lock_tensor_buffer = reinterpret_cast<LockTensorBufferFn>(
      dlsym(lib, "LiteRtLockTensorBuffer"));
  unlock_tensor_buffer = reinterpret_cast<UnlockTensorBufferFn>(
      dlsym(lib, "LiteRtUnlockTensorBuffer"));
  destroy_tensor_buffer = reinterpret_cast<DestroyTensorBufferFn>(
      dlsym(lib, "LiteRtDestroyTensorBuffer"));
  run_compiled_model = reinterpret_cast<RunCompiledModelFn>(
      dlsym(lib, "LiteRtRunCompiledModel"));

  jclass jni_handle_class =
      env->FindClass("com/google/ai/edge/litert/JniHandle");
  if (jni_handle_class == nullptr) {
    return false;
  }
  jni_handle_field = env->GetFieldID(jni_handle_class, "handle", "J");

  return resize_input_tensor != nullptr &&
         get_output_tensor_layouts != nullptr && get_environment != nullptr &&
         create_managed_tensor_buffer != nullptr &&
         lock_tensor_buffer != nullptr && unlock_tensor_buffer != nullptr &&
         destroy_tensor_buffer != nullptr && run_compiled_model != nullptr &&
         jni_handle_field != nullptr;
}

const JNINativeMethod kMethods[] = {
    {"nativeResizeInputTensor",
     "(Lcom/google/ai/edge/litert/CompiledModel;II[I)V",
     reinterpret_cast<void*>(ResizeInputTensor)},
    {"nativeUpdateOutputLayouts",
     "(Lcom/google/ai/edge/litert/CompiledModel;II)[[I",
     reinterpret_cast<void*>(UpdateOutputLayouts)},
    {"nativeRunDynamic",
     "(Lcom/google/ai/edge/litert/CompiledModel;I[[F[[I[[I)[[F",
     reinterpret_cast<void*>(RunDynamic)},
};

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  if (!Resolve(env)) {
    return JNI_ERR;
  }
  jclass binding_class = env->FindClass(
      "com/google/ai/edge/examples/text_to_speech_streaming/"
      "LiteRtDynamicShape");
  if (binding_class == nullptr ||
      env->RegisterNatives(binding_class, kMethods, 3) != JNI_OK) {
    return JNI_ERR;
  }
  return JNI_VERSION_1_6;
}
