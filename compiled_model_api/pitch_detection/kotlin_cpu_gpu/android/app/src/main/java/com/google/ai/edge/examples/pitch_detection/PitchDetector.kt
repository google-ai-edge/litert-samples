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

package com.google.ai.edge.examples.pitch_detection

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.log2
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * CREPE monophonic pitch (f0) detection (marl/crepe, MIT), on-device, fully on the LiteRT
 * CompiledModel GPU (ML Drift).
 *
 *   frame[1,1024] (16 kHz, per-frame zero-mean/unit-var) --[GPU CNN]--> activations[1,360] (pitch bins)
 *
 * The whole network is a pure CNN — 6× {zero-pad, Conv2d, ReLU, BatchNorm, MaxPool} + Linear +
 * sigmoid — so it rides the GPU graph at corr 1.0 vs PyTorch in fp16. Per-frame normalization and
 * the 360-bin → cents → Hz decode are host-side. The 44.5 MB tflite is loaded from filesDir
 * (pushed via install_to_device.sh).
 */
class PitchDetector(context: Context) : Closeable {

  companion object {
    const val MODEL = "crepe_full_fp16.tflite"
    const val SAMPLE_RATE = 16000
    const val WINDOW = 1024
    const val BINS = 360
    const val CENTS_PER_BIN = 20.0
    const val CENTS_OFFSET = 1997.3794084376191   // torchcrepe bins→cents intercept
    private val NOTE_NAMES = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
  }

  private val model: CompiledModel = run {
    val f = File(context.filesDir, MODEL)
    check(f.exists()) { "Model not found: $MODEL. Push it first:\n  ./install_to_device.sh" }
    CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  }
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  /** hz: detected pitch (Hz); confidence: peak bin activation 0..1; note/cents: nearest note + offset. */
  data class Pitch(val hz: Float, val confidence: Float, val note: String, val octave: Int, val cents: Int)

  /** Detect pitch from one WINDOW-sample 16 kHz frame. Normalizes per-frame, runs the GPU net, decodes. */
  fun detect(frame: FloatArray): Pitch {
    val x = FloatArray(WINDOW)
    var mean = 0f
    for (i in 0 until WINDOW) {
      mean += frame[i]
    }
    mean /= WINDOW
    var v = 0f
    for (i in 0 until WINDOW) {
        val d = frame[i] - mean
        v += d * d
    }
    val std = maxOf(sqrt(v / WINDOW), 1e-10f)
    for (i in 0 until WINDOW) {
      x[i] = (frame[i] - mean) / std
    }

    inBuf[0].writeFloat(x)
    model.run(inBuf, outBuf)
    val act = outBuf[0].readFloat()                 // [360]

    // weighted average of cents over ±4 bins around the peak (torchcrepe 'weighted_argmax')
    var c = 0
    for (i in 1 until BINS) {
      if (act[i] > act[c]) {
        c = i
      }
    }
    val s = maxOf(0, c - 4)
    val e = minOf(BINS, c + 5)
    var num = 0.0
    var den = 0.0
    for (i in s until e) {
        num += act[i].toDouble() * i
        den += act[i].toDouble()
    }
    val cents = CENTS_PER_BIN * (num / den) + CENTS_OFFSET
    val hz = (10.0 * Math.pow(2.0, cents / 1200.0)).toFloat()

    val midi = 69.0 + 12.0 * log2(hz / 440.0)
    val nearest = midi.roundToInt()
    val centsOff = ((midi - nearest) * 100).roundToInt()
    val name = NOTE_NAMES[((nearest % 12) + 12) % 12]
    return Pitch(hz, act[c], name, nearest / 12 - 1, centsOff)
  }

  override fun close() {
    inBuf.forEach { it.close() }
    outBuf.forEach { it.close() }
    model.close()
  }
}
