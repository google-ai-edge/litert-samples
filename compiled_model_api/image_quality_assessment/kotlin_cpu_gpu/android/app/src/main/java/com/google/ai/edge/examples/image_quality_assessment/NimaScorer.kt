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

package com.google.ai.edge.examples.image_quality_assessment

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * NIMA (Neural Image Assessment) image-quality scoring on the LiteRT CompiledModel GPU. Two MobileNet
 * models — aesthetic (AVA) and technical (TID2013) — each output a 10-bin score distribution; the
 * quality score is the distribution mean over 1..10. Both models run fully on the GPU.
 */
class NimaScorer(ctx: Context) {

  companion object { const val SIZE = 224 }

  private val aesthetic = CompiledModel.create(ctx.assets, "nima_aesthetic_fp16.tflite", CompiledModel.Options(Accelerator.GPU), null)
  private val aIn = aesthetic.createInputBuffers()
  private val aOut = aesthetic.createOutputBuffers()
  private val technical = CompiledModel.create(ctx.assets, "nima_technical_fp16.tflite", CompiledModel.Options(Accelerator.GPU), null)
  private val tIn = technical.createInputBuffers()
  private val tOut = technical.createOutputBuffers()

  /** MobileNet preprocessing: resize 224², RGB, NHWC, scaled to [-1, 1]. */
  private fun preprocess(bm: Bitmap): FloatArray {
    val s = Bitmap.createScaledBitmap(bm, SIZE, SIZE, true)
    val px = IntArray(SIZE * SIZE)
    s.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
    val out = FloatArray(SIZE * SIZE * 3)
    for (i in 0 until SIZE * SIZE) {
      val p = px[i]
      out[i * 3] = (p shr 16 and 0xFF) / 127.5f - 1f
      out[i * 3 + 1] = (p shr 8 and 0xFF) / 127.5f - 1f
      out[i * 3 + 2] = (p and 0xFF) / 127.5f - 1f
    }
    return out
  }

  private fun meanScore(dist: FloatArray): Float {
    var s = 0f
    for (i in dist.indices) {
        s += (i + 1) * dist[i]
    }
    return s
  }

  data class Scores(val aesthetic: Float, val technical: Float)

  fun score(bm: Bitmap): Scores {
    val x = preprocess(bm)
    aIn[0].writeFloat(x)
    aesthetic.run(aIn, aOut)
    tIn[0].writeFloat(x)
    technical.run(tIn, tOut)
    return Scores(meanScore(aOut[0].readFloat()), meanScore(tOut[0].readFloat()))
  }

  fun close() {
      aesthetic.close()
      technical.close()
  }
}
