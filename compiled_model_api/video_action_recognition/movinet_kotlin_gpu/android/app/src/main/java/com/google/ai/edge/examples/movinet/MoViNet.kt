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

package com.google.ai.edge.examples.movinet

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * MoViNet-A0 streaming video action recognition on LiteRT CompiledModel (GPU).
 *
 * The causal streaming model recognises an action across a sequence of frames,
 * one frame at a time with constant memory. It is re-authored so every tensor is
 * 4D and all recurrent state is threaded through the graph I/O:
 *
 *   input[0]       = frame  [1, 3, 172, 172]  NCHW, RGB, 0..1
 *   input[1..28]   = 28 temporal-conv stream-buffer frames [1,C,H,W] (oldest first)
 *   input[29..44]  = 16 pooling running-sums               [1,C,1,1]
 *   input[45]      = inv_count = 1 / current frame number  [1,1,1,1]
 *   input[46]      = constant 1.0                          [1,1,1,1]
 *   output[0]      = logits [1, 600]  (Kinetics-600)
 *   output[1..11]  = current per-temporal-conv frame       [1,C,H,W]
 *   output[12..27] = 16 fresh per-frame spatial means       [1,C,1,1]
 *
 * The stream-buffer shift register and the pool running-sum accumulation are both
 * done host-side (below): the graph consumes the recurrent state but only emits
 * fresh tensors. This side-steps three quirks of the Mali GPU delegate for
 * stateful graphs — an input passed through to an output loses its compute-side
 * use; a `state + tensor` output is zeroed; and a conv-output tensor that is both
 * consumed and emitted has its emitted copy corrupted (so each emitted stream
 * frame is decoupled from its compute use by a multiply against the `1.0` input).
 */
class MoViNet(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "MoViNet"
    const val INPUT_SIZE = 172
    private val STREAM_DIMS = intArrayOf(2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4)
    private const val N_POOL = 16
    private const val INV_COUNT_IN = 45
    private const val ONE_IN = 46
  }

  private val model: CompiledModel = CompiledModel.create(
    modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val streamOffset = IntArray(STREAM_DIMS.size)
  private val poolSums = Array(N_POOL) { FloatArray(0) }
  private var frameCount = 0f

  init {
    Log.i(TAG, "GPU compiled OK — ${inBufs.size} inputs / ${outBufs.size} outputs")
    var off = 1
    for (c in STREAM_DIMS.indices) { streamOffset[c] = off; off += STREAM_DIMS[c] }
    for (i in 0 until N_POOL) poolSums[i] = FloatArray(inBufs[29 + i].readFloat().size)
    inBufs[ONE_IN].writeFloat(floatArrayOf(1f))
    reset()
  }

  /** Zero all recurrent state (start a fresh clip). */
  fun reset() {
    for (k in 1..28) inBufs[k].writeFloat(FloatArray(inBufs[k].readFloat().size))
    for (i in 0 until N_POOL) {
      java.util.Arrays.fill(poolSums[i], 0f)
      inBufs[29 + i].writeFloat(poolSums[i])
    }
    frameCount = 0f
  }

  /** Run one streaming frame (NCHW [3*172*172], RGB, 0..1). Returns [600] logits. */
  fun classify(frameNCHW: FloatArray): FloatArray {
    frameCount += 1f
    inBufs[0].writeFloat(frameNCHW)
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

  override fun close() = model.close()
}
