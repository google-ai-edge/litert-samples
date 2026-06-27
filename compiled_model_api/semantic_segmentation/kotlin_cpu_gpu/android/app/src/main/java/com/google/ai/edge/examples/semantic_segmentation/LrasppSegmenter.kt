/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.semantic_segmentation

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * LR-ASPP MobileNetV3-Large semantic segmentation on the LiteRT CompiledModel GPU.
 *   image[1,3,512,512] (ImageNet-normalized) -> logits[1,512,512,21] (NHWC, 21 COCO-VOC classes)
 *
 * Pure CNN (MobileNetV3 backbone + Lite R-ASPP head). The only GPU re-authoring is the 9
 * `AdaptiveAvgPool2d(1)` global pools (SE blocks + the R-ASPP scale branch) -> two single-axis means.
 * ~5 ms / 512x512 on a Pixel 8a; device output == PyTorch (argmax agreement 99.85%). The per-pixel
 * argmax (class id) runs here on the CPU - trivial.
 */
class LrasppSegmenter(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

  companion object {
    const val SIZE = 512
    const val CLASSES = 21
    const val MODEL = "lraspp_fp16.tflite"
    // ImageNet normalization (torchvision segmentation preprocessing)
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model: CompiledModel = run {
    val f = File(ctx.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push first with install_to_device.sh" }
    CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
  }
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns [SIZE*SIZE] per-pixel class ids (0..20). */
  fun segment(rgb: FloatArray): IntArray {
    val hw = SIZE * SIZE
    val chw = FloatArray(3 * hw)
    for (i in 0 until hw) {
      chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
      chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
      chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
    }
    inBuf[0].writeFloat(chw)
    model.run(inBuf, outBuf)
    val logits = outBuf[0].readFloat()        // [SIZE*SIZE*CLASSES] NHWC
    val cls = IntArray(hw)
    for (p in 0 until hw) {
      val base = p * CLASSES
      var best = 0; var bestV = logits[base]
      for (c in 1 until CLASSES) {
        val v = logits[base + c]
        if (v > bestV) { bestV = v; best = c }
      }
      cls[p] = best
    }
    return cls
  }

  override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
