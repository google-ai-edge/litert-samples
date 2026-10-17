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

package android.template.ui.classify

import android.graphics.Bitmap

/** The photo as the model's input: scaled to 224 x 224, values from 0 to 1. */
fun preprocess(bitmap: Bitmap): FloatArray {
    val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val pixels = IntArray(224 * 224)
    scaled.getPixels(pixels, 0, 224, 0, 0, 224, 224)
    val input = FloatArray(224 * 224)
    pixels.forEachIndexed { i, p ->
        val r = p shr 16 and 0xFF
        val g = p shr 8 and 0xFF
        val b = p and 0xFF
        input[i] = (r + g + b) / 3f / 255f
    }
    return input
}
