// Minimal LiteRT C API declarations for the SAM2 iOS Metal benchmark app.
// Symbols are resolved at link time from the LiteRTLM xcframework.
#ifndef SAM2_BRIDGING_HEADER_H_
#define SAM2_BRIDGING_HEADER_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtModelT* LiteRtModel;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtCompiledModelT* LiteRtCompiledModel;
typedef struct LiteRtTensorBufferT* LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;

typedef size_t LiteRtParamIndex;
typedef int LiteRtHwAcceleratorSet;

typedef enum {
  kLiteRtStatusOk = 0,
} LiteRtStatus;

typedef enum LiteRtHwAccelerators {
  kLiteRtHwAcceleratorNone = 0,
  kLiteRtHwAcceleratorCpu = 1 << 0,
  kLiteRtHwAcceleratorGpu = 1 << 1,
  kLiteRtHwAcceleratorNpu = 1 << 2,
  _kLiteRtHwAcceleratorNegativeDummy = -1,
} LiteRtHwAccelerators;

typedef enum {
  kLiteRtTensorBufferLockModeRead = 0,
  kLiteRtTensorBufferLockModeWrite = 1,
  kLiteRtTensorBufferLockModeReadWrite = 2,
} LiteRtTensorBufferLockMode;

typedef enum {
  kLiteRtTensorBufferTypeUnknown = 0,
  kLiteRtTensorBufferTypeHostMemory = 1,
} LiteRtTensorBufferType;

typedef enum {
  kLiteRtElementTypeNone = 0,
  kLiteRtElementTypeFloat32 = 1,
} LiteRtElementType;

#define LITERT_TENSOR_MAX_RANK 8

typedef struct {
  unsigned int rank : 7;
  bool has_strides : 1;
  int32_t dimensions[LITERT_TENSOR_MAX_RANK];
  uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

typedef struct {
  LiteRtElementType element_type;
  LiteRtLayout layout;
} LiteRtRankedTensorType;

typedef struct LiteRtEnvOption LiteRtEnvOption;

LiteRtStatus LiteRtCreateEnvironment(int num_options,
                                     const LiteRtEnvOption* options,
                                     LiteRtEnvironment* environment);
void LiteRtDestroyEnvironment(LiteRtEnvironment environment);

LiteRtStatus LiteRtCreateModelFromFile(LiteRtEnvironment environment,
                                       const char* filename,
                                       LiteRtModel* model);
void LiteRtDestroyModel(LiteRtModel model);

LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options);
void LiteRtDestroyOptions(LiteRtOptions options);
LiteRtStatus LiteRtSetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet hardware_accelerators);

LiteRtStatus LiteRtCreateCompiledModel(LiteRtEnvironment environment,
                                       LiteRtModel model,
                                       LiteRtOptions compilation_options,
                                       LiteRtCompiledModel* compiled_model);
void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model);
LiteRtStatus LiteRtCompiledModelIsFullyAccelerated(
    LiteRtCompiledModel compiled_model, bool* fully_accelerated);

LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements);
LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements);
LiteRtStatus LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
    LiteRtTensorBufferRequirements requirements, int type_index,
    LiteRtTensorBufferType* type);
LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size);

LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* buffer);
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer buffer);
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr,
                                    LiteRtTensorBufferLockMode lock_mode);
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer buffer);

LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    LiteRtParamIndex signature_index,
                                    size_t num_input_buffers,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_output_buffers,
                                    LiteRtTensorBuffer* output_buffers);

// Build a float32 ranked tensor type from a dimensions array without exposing
// C bitfields (`rank : 7`) to Swift.
static inline LiteRtRankedTensorType Sam2MakeFloat32Type(const int32_t* dims,
                                                         unsigned int rank) {
  LiteRtRankedTensorType type;
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = rank;
  type.layout.has_strides = false;
  for (unsigned int i = 0; i < rank; ++i) {
    type.layout.dimensions[i] = dims[i];
  }
  return type;
}

#ifdef __cplusplus
}
#endif

#endif  // SAM2_BRIDGING_HEADER_H_
