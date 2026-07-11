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

package com.google.ai.edge.examples.plantnet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * PlantNet-300K fine-grained plant species classifier on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 224, 224]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1081]         species logits (PlantNet-300K, Latin names).
 *
 * A torchvision ResNet18 (pure CNN). One re-authoring patch (baked into the graph,
 * see conversion/): the ResNet stem MaxPool's -inf-pad PADV2 is replaced with a
 * 0-pad + unpadded maxpool (exact post-ReLU), which the Mali delegate accepts.
 */
class PlantClassifier(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "PlantNet"
    const val SIZE = 224
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Classify. Returns top-[topK] (species name, probability) + time (ms). */
  fun classify(bitmap: Bitmap, topK: Int = 5): Pair<List<Pair<String, Float>>, Long> {
    val t = System.nanoTime()
    val side = minOf(bitmap.width, bitmap.height)
    matrix.reset()
    matrix.postTranslate(-(bitmap.width - side) / 2f, -(bitmap.height - side) / 2f)
    matrix.postScale(SIZE.toFloat() / side, SIZE.toFloat() / side)
    Canvas(resized).drawBitmap(bitmap, matrix, paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val logits = outBufs[0].readFloat()   // [1081]

    val idx = logits.indices.sortedByDescending { logits[it] }.take(topK)
    val mx = logits[idx.first()]
    var sum = 0.0
    for (v in logits) {
      sum += Math.exp((v - mx).toDouble())
    }
    val preds = idx.map { i ->
      PlantNetLabels.NAMES[i] to (Math.exp((logits[i] - mx).toDouble()) / sum).toFloat()
    }
    return preds to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}
