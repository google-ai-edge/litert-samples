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

package com.google.ai.edge.examples.image_cutout

/** IEEE 754 binary16 to float32 (exact: every fp16 value is a float32 value). */
object Half {
  fun toFloat(bits: Int): Float {
    val h = bits and 0xFFFF
    val sign = (h ushr 15) and 1
    val exp = (h ushr 10) and 0x1F
    val mant = h and 0x3FF
    val magnitude =
      when (exp) {
        0 -> mant * SUBNORMAL_UNIT // zero and subnormals: mant * 2^-24, exact in float32
        0x1F -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
        else -> java.lang.Float.intBitsToFloat(((exp - 15 + 127) shl 23) or (mant shl 13))
      }
    return if (sign == 1) -magnitude else magnitude
  }

  private val SUBNORMAL_UNIT = java.lang.Float.intBitsToFloat((127 - 24) shl 23) // 2^-24
}
