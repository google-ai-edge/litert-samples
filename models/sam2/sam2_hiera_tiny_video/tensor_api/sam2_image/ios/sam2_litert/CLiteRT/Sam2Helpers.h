// Small C helpers so Swift never has to construct LiteRtLayout's C bitfields
// (`rank : 7`) directly — the C compiler fills them here.
#ifndef SAM2_HELPERS_H_
#define SAM2_HELPERS_H_

#include <stdbool.h>
#include <stdint.h>

#include "litert/c/litert_model_types.h"

// Build a float32 ranked tensor type from a dimensions array.
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

#endif  // SAM2_HELPERS_H_
