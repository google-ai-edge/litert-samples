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

package com.google.ai.edge.examples.model_zoo.models.movinet

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

/**
 * MoViNet-A0 streaming video action recognition on LiteRT CompiledModel (GPU).
 *
 * Graph I/O (see scripts/stream_model.py): input[0] = frame [1, 3, 172, 172] NCHW, RGB, 0..1
 * input[1..28] = 28 temporal-conv stream-buffer frames [1,C,H,W] (11 convs, dim_pad in {2,4},
 * oldest first) input[29..44] = 16 pooling running-sums [1,C,1,1] input[45] = frame counter
 * (current frame number, >= 1) output[0] = logits [1, 600] (Kinetics-600) output[1..11] = current
 * per-temporal-conv frame [1,C,H,W] output[12..27] = 16 fresh per-frame spatial means [1,C,1,1]
 *
 * The stream-buffer shift register AND the pooling running-sum accumulation are done HOST-SIDE. The
 * graph only consumes the recurrent state and emits fresh tensors — this side-steps two Mali GPU
 * delegate bugs (an input passed through to an output loses its compute-side use; a `state +
 * tensor` output is zeroed).
 */
class ActionRecognizer(
  context: Context,
  modelFile: File,
  accelerator: Accelerator = Accelerator.GPU,
  private val windowFrames: Int = 64,
) : AutoCloseable {

  companion object {
    private const val TAG = "MoViNet"
    const val INPUT_SIZE = 172
    // temporal-conv buffer depths (dim_pad) in STREAM_SPEC order
    private val STREAM_DIMS = intArrayOf(2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4)
    private const val N_POOL = 16
    private const val INV_COUNT_IN = 45 // 1 / frame number
    private const val ONE_IN = 46 // constant 1.0 (output decoupler)

    internal fun topK(logits: FloatArray, k: Int): List<Prediction> {
      val idx = (logits.indices).sortedByDescending { logits[it] }.take(k)
      val maxLogit = logits[idx.first()]
      var sumExp = 0.0
      for (v in logits) sumExp += Math.exp((v - maxLogit).toDouble())
      return idx.map { i ->
        Prediction(
          i,
          Kinetics600Labels.NAMES[i],
          (Math.exp((logits[i] - maxLogit).toDouble()) / sumExp).toFloat(),
        )
      }
    }
  }

  private val model: CompiledModel
  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  /** Start index into inBufs for each temporal conv's stream frames. */
  private val streamOffset = IntArray(STREAM_DIMS.size)
  private val poolSums = Array(N_POOL) { FloatArray(0) }
  private var frameCount = 0f

  private val inputFloats = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
  private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
  private val resized = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    val options = CompiledModel.Options(accelerator)
    model = CompiledModel.create(modelFile.absolutePath, options, null)
    inBufs = model.createInputBuffers()
    outBufs = model.createOutputBuffers()
    Log.i(TAG, "$accelerator compiled OK — ${inBufs.size} inputs / ${outBufs.size} outputs")

    var off = 1
    for (c in STREAM_DIMS.indices) {
      streamOffset[c] = off
      off += STREAM_DIMS[c]
    }
    for (i in 0 until N_POOL) poolSums[i] = FloatArray(inBufs[29 + i].readFloat().size)
    inBufs[ONE_IN].writeFloat(floatArrayOf(1f)) // constant decoupler input
    reset()
  }

  /** Zero all recurrent state (restart the classification window). */
  fun reset() {
    for (k in 1..28) inBufs[k].writeFloat(FloatArray(inBufs[k].readFloat().size))
    for (i in 0 until N_POOL) {
      java.util.Arrays.fill(poolSums[i], 0f)
      inBufs[29 + i].writeFloat(poolSums[i])
    }
    frameCount = 0f
  }

  /** Run one streaming frame given the pre-filled input[0]. Updates all state. */
  private fun runFrame(): FloatArray {
    frameCount += 1f
    inBufs[INV_COUNT_IN].writeFloat(floatArrayOf(1f / frameCount))
    model.run(inBufs, outBufs)
    val logits = outBufs[0].readFloat()

    // stream buffers: shift register (drop oldest, append current) host-side
    for (c in STREAM_DIMS.indices) {
      val s = outBufs[1 + c].readFloat()
      val base = streamOffset[c]
      val dp = STREAM_DIMS[c]
      for (i in 0 until dp - 1) inBufs[base + i].writeFloat(inBufs[base + i + 1].readFloat())
      inBufs[base + dp - 1].writeFloat(s)
    }
    // pooling: accumulate running sum host-side
    for (i in 0 until N_POOL) {
      val mean = outBufs[12 + i].readFloat()
      val sum = poolSums[i]
      for (j in sum.indices) sum[j] += mean[j]
      inBufs[29 + i].writeFloat(sum)
    }
    return logits
  }

  /** One streaming frame from a camera bitmap. Returns running top-[topK]. */
  fun classify(bitmap: Bitmap, topK: Int = 5): Pair<List<Prediction>, Long> {
    val t = System.nanoTime()
    val canvas = Canvas(resized)
    matrix.setScale(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
    canvas.drawBitmap(bitmap, matrix, paint)
    resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
    val plane = INPUT_SIZE * INPUT_SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 255f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
      inputFloats[2 * plane + i] = (p and 0xFF) / 255f
    }
    inBufs[0].writeFloat(inputFloats)
    val logits = runFrame()
    val preds = topK(logits, topK)
    val ms = (System.nanoTime() - t) / 1_000_000
    if (frameCount >= windowFrames) reset()
    return preds to ms
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}

data class Prediction(val index: Int, val label: String, val score: Float)
