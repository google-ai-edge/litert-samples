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

package com.google.ai.edge.examples.metric_depth

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Metric3D v2 (ViT-Small) monocular METRIC depth on the LiteRT CompiledModel GPU.
 *   image[1,3,448,448] (ImageNet-normalized) -> depth[1,1,448,448] (absolute depth in meters,
 *   in the canonical camera space the model was trained on).
 *
 * The DINOv2 ViT-S encoder + RAFT-DPT decoder run entirely on the GPU (CompiledModel / ML Drift):
 * the RAFT convex upsample is re-authored as a depth-to-space ZeroStuffConvT2d and GELU as the
 * accurate tanh approximation so every op is GPU-compatible. ~44 ms / inference on a Pixel 8a.
 *
 * Metric scale note: the model outputs depth for a canonical camera (focal 1000 at the canonical
 * resolution). For a calibrated camera, multiply by (fx / 1000) — the de-canonical transform — using
 * the real focal length. With no intrinsics we show the raw canonical-metric depth (already in meters).
 */
class MetricDepth(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

  companion object {
    const val SIZE = 448
    const val MODEL = "metric3d_fp16.tflite"
    // ImageNet normalization in 0-255 scale (Metric3D preprocessing).
    val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
    val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
  }

  private val model: CompiledModel = run {
    val f = File(ctx.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push first with install_to_device.sh" }
    CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
  }
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns [SIZE*SIZE] depth in meters (row-major). */
  fun depth(rgb: FloatArray): FloatArray {
    val hw = SIZE * SIZE
    val chw = FloatArray(3 * hw)
    for (i in 0 until hw) {
      chw[i] = (rgb[i * 3] - MEAN[0]) / STD[0]
      chw[hw + i] = (rgb[i * 3 + 1] - MEAN[1]) / STD[1]
      chw[2 * hw + i] = (rgb[i * 3 + 2] - MEAN[2]) / STD[2]
    }
    inBuf[0].writeFloat(chw)
    model.run(inBuf, outBuf)
    return outBuf[0].readFloat()           // [SIZE*SIZE] depth (meters)
  }

  override fun close() {
    inBuf.forEach { it.close() }
    outBuf.forEach { it.close() }
    model.close()
  }
}
