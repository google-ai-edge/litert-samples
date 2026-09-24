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

package com.google.ai.edge.examples.model_zoo.models.plantnet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import org.json.JSONObject

/**
 * PlantNet-300K fine-grained plant species classifier on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 224, 224] NCHW, RGB, ImageNet-normalized. Output: [1, 1081] species logits
 * (PlantNet-300K, Latin names).
 *
 * A torchvision ResNet18 — pure CNN. One re-authoring patch (baked into the graph, see scripts/):
 * the ResNet stem MaxPool's -inf-pad PADV2 is replaced with a 0-pad + unpadded maxpool (exact
 * post-ReLU), which the Mali delegate accepts.
 */
class PlantClassifier(
  context: Context,
  modelFile: File,
  labelsFile: File,
  accelerator: Accelerator = Accelerator.GPU,
) : AutoCloseable {

  companion object {
    internal fun decode(
      logits: FloatArray,
      names: Array<String>,
      topK: Int,
    ): List<Pair<String, Float>> {
      val idx = logits.indices.sortedByDescending { logits[it] }.take(topK)
      val mx = logits[idx.first()]
      var sum = 0.0
      for (v in logits) sum += Math.exp((v - mx).toDouble())
      val preds =
        idx.map { i -> names[i] to (Math.exp((logits[i] - mx).toDouble()) / sum).toFloat() }
      return preds
    }

    private const val TAG = "PlantNet"
    const val SIZE = 224
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val labels =
    JSONObject(labelsFile.readText()).let { json ->
      json.keys().asSequence().toList().sorted().map { json.getString(it) }.toTypedArray()
    }
  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    val options = CompiledModel.Options(accelerator)
    model = CompiledModel.create(modelFile.absolutePath, options, null)
    inBufs = model.createInputBuffers()
    outBufs = model.createOutputBuffers()
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Classify. Returns top-[topK] (species name, probability) + time (ms). */
  fun classify(bitmap: Bitmap, topK: Int = 5): Pair<List<Pair<String, Float>>, Long> {
    val t = System.nanoTime()
    // center-crop to square, resize to 224
    val side = minOf(bitmap.width, bitmap.height)
    val sx = (bitmap.width - side) / 2f
    val sy = (bitmap.height - side) / 2f
    val canvas = Canvas(resized)
    matrix.reset()
    matrix.postTranslate(-sx, -sy)
    matrix.postScale(SIZE.toFloat() / side, SIZE.toFloat() / side)
    canvas.drawBitmap(bitmap, matrix, paint)
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
    val logits = outBufs[0].readFloat() // [1081]

    val preds = decode(logits, labels, topK)
    return preds to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}
