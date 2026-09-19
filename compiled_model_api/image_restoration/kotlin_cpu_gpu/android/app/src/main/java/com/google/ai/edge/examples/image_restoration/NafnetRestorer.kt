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

package com.google.ai.edge.examples.image_restoration

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * NAFNet (Nonlinear Activation Free Network) image restoration on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (RGB, [0,1]) -> restored[1,3,256,256] (RGB, [0,1])
 *
 * NAFNet-GoPro-width32 removes motion blur. A pure CNN with no activation functions (SimpleGate =
 * channel-split multiply), so the whole U-Net rides the GPU. The LayerNorm is re-authored fp16-safe
 * (the deep residual stream overflows the fp16 channel-sum on Mali otherwise) and the PixelShuffle
 * upsamples as a depth-to-space ZeroStuffConvT2d. ~42 ms / 256x256 on a Pixel 8a; device output ==
 * PyTorch (corr 1.0).
 */
class NafnetRestorer(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

  companion object {
    const val SIZE = 256
    const val MODEL = "nafnet_fp16.tflite"
  }

  private val model: CompiledModel = run {
    val f = File(ctx.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push first with install_to_device.sh" }
    CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
  }
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns the restored SIZE*SIZE*3 row-major [0,255]. */
  fun restore(rgb: FloatArray): FloatArray {
    val hw = SIZE * SIZE
    val chw = FloatArray(3 * hw)
    for (i in 0 until hw) {                       // HWC [0,255] -> CHW [0,1]
      chw[i] = rgb[i * 3] / 255f
      chw[hw + i] = rgb[i * 3 + 1] / 255f
      chw[2 * hw + i] = rgb[i * 3 + 2] / 255f
    }
    inBuf[0].writeFloat(chw)
    model.run(inBuf, outBuf)
    val out = outBuf[0].readFloat()               // CHW [0,1]
    val rgbOut = FloatArray(3 * hw)
    for (i in 0 until hw) {
      rgbOut[i * 3] = (out[i] * 255f).coerceIn(0f, 255f)
      rgbOut[i * 3 + 1] = (out[hw + i] * 255f).coerceIn(0f, 255f)
      rgbOut[i * 3 + 2] = (out[2 * hw + i] * 255f).coerceIn(0f, 255f)
    }
    return rgbOut
  }

  override fun close() {
      inBuf.forEach { it.close() }
      outBuf.forEach { it.close() }
      model.close()
  }
}
