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

package com.google.ai.edge.examples.model_zoo.models.dac

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Descript Audio Codec (DAC 16 kHz), on-device. The two convolutional halves run on the LiteRT
 * CompiledModel GPU (ML Drift); the RVQ (codes <-> latent) runs on CPU in [DacRVQ].
 *
 * audio[16000] -> encoder.tflite (GPU) -> z[1,1024,50] -> RVQ.encode -> codes[12,50] -> RVQ.decode
 * -> z_q[1,1024,50] -> decoder.tflite (GPU) -> audio[~15992]
 *
 * The decoder's ConvTranspose1d were rewritten to a GPU-clean zero-stuff form (ConvTranspose1d /
 * TRANSPOSE_CONV are rejected by Mali), so the whole conv graph stays on the GPU delegate. Both
 * tflites are loaded from filesDir (push via scripts/install_to_device.sh).
 */
class DacCodec(private val modelDir: File, private val accelerator: Accelerator) : Closeable {

  companion object {
    const val SAMPLES = 16000 // 1 s @ 16 kHz
    const val HID = 1024
    const val T = 50 // code frames (hop 320)
    const val ENCODER = "dac_16khz_encoder_fp16.tflite"
    const val DECODER = "dac_16khz_deconly_zs_fp16.tflite"
  }

  private fun load(name: String): CompiledModel {
    val f = File(modelDir, name)
    check(f.exists()) { "Model not found: $name. Push it first:\n  scripts/install_to_device.sh" }
    return CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
  }

  private val encoder = load(ENCODER)
  private val decoder =
    try {
      load(DECODER)
    } catch (failure: Exception) {
      encoder.close()
      throw failure
    }
  private val rvq = DacRVQ(File(modelDir, "dac_rvq.bin").readBytes())
  private val encIn = encoder.createInputBuffers()
  private val encOut = encoder.createOutputBuffers()
  private val decIn = decoder.createInputBuffers()
  private val decOut = decoder.createOutputBuffers()

  data class Result(
    val audio: FloatArray,
    val codes: IntArray,
    val encodeMs: Long,
    val decodeMs: Long,
  )

  /** Full encode -> quantize -> decode round-trip of a 16000-sample clip. */
  fun roundTrip(audio: FloatArray): Result {
    val t0 = System.nanoTime()
    encIn[0].writeFloat(audio)
    encoder.run(encIn, encOut)
    val z = encOut[0].readFloat()
    val codes = rvq.encode(z, T)
    val t1 = System.nanoTime()
    val zq = rvq.decode(codes, T)
    decIn[0].writeFloat(zq)
    decoder.run(decIn, decOut)
    val out = decOut[0].readFloat()
    val t2 = System.nanoTime()
    return Result(out, codes, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000)
  }

  override fun close() {
    encIn.forEach { it.close() }
    encOut.forEach { it.close() }
    decIn.forEach { it.close() }
    decOut.forEach { it.close() }
    encoder.close()
    decoder.close()
  }
}
