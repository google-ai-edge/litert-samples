// C helper so Swift never constructs LiteRtLayout's C bitfields (`rank : 7`)
// directly — the C compiler fills them here. Same pattern as SAM2's helper,
// generalized to any element type (the text encoder takes int32 ids/mask).
#ifndef BONSAI_HELPERS_H_
#define BONSAI_HELPERS_H_

#include <stdint.h>

#include "litert/c/litert_model_types.h"

static inline LiteRtRankedTensorType BonsaiMakeType(LiteRtElementType elem,
                                                    const int32_t* dims,
                                                    unsigned int rank) {
  LiteRtRankedTensorType type;
  type.element_type = elem;
  type.layout.rank = rank;
  type.layout.has_strides = false;
  for (unsigned int i = 0; i < rank; ++i) {
    type.layout.dimensions[i] = dims[i];
  }
  return type;
}

// ---------------------------------------------------------------------------
// Classic TFLite C API — the CLiteRTLM.framework binary exports these symbols
// (verified with nm) but ships no headers for them. The Next-API CompiledModel
// CPU path has no public thread knob in v0.13.1 and runs single-threaded
// (measured: DiT 38.8 s/step); this path takes SetNumThreads.
// ---------------------------------------------------------------------------
#include <stddef.h>

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;

extern TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
extern void TfLiteModelDelete(TfLiteModel* model);
extern TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate(void);
extern void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options);
extern void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options,
                                                  int32_t num_threads);
extern TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model, const TfLiteInterpreterOptions* options);
extern void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
extern int32_t TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter);
extern int32_t TfLiteInterpreterGetInputTensorCount(const TfLiteInterpreter* interpreter);
extern TfLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter* interpreter,
                                                     int32_t input_index);
extern int32_t TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter);
extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index);
extern const char* TfLiteTensorName(const TfLiteTensor* tensor);
extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
extern int32_t TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor, const void* input_data,
                                          size_t input_data_size);
extern int32_t TfLiteTensorCopyToBuffer(const TfLiteTensor* output_tensor,
                                        void* output_data, size_t output_data_size);

// The classic path does NOT auto-apply XNNPACK in this build (measured: the
// reference kernels ran the int4 text encoder in 1676 s with wrong numerics).
// Attach the delegate explicitly, with the thread count in its options.
//
// The options struct below is copied VERBATIM from the real header
// (litert-tensor/LiteRT/tflite/delegates/xnnpack/xnnpack_delegate.h,
// checkout 2026-07-22, >= the v0.13.1 binary). We only ever WRITE
// num_threads — offset 0, unchanged across versions; every other field is
// produced and consumed by the same library version via OptionsDefault(), so
// interior field drift between header and binary cannot bite. (Do NOT pass a
// pthreadpool as the second arg of CreateWithThreadpool: that parameter is a
// TfLiteContext*, and the mistake is a SIGSEGV at startup.)
#include <stdbool.h>

typedef struct TfLiteDelegate TfLiteDelegate;
struct TfLiteXNNPackDelegateWeightsCache;

typedef struct {
  int32_t num_threads;
  uint32_t runtime_flags;
  uint32_t flags;
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
  bool handle_variable_ops;
  const char* weight_cache_file_path;
  int weight_cache_file_descriptor;
  void* weight_cache_provider;
  bool weight_cache_lock_memory;
} TfLiteXNNPackDelegateOptions;

extern TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault(void);
extern TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options);
extern void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);
extern void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* options,
                                                TfLiteDelegate* delegate);

static inline TfLiteDelegate* BonsaiCreateXnnpackDelegate(int32_t threads) {
  TfLiteXNNPackDelegateOptions o = TfLiteXNNPackDelegateOptionsDefault();
  o.num_threads = threads;
  return TfLiteXNNPackDelegateCreate(&o);
}

#endif  // BONSAI_HELPERS_H_
