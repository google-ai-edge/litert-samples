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

package com.google.ai.edge.examples.speaker_diarization

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/**
 * WeSpeaker ResNet34 speaker embedding (CC-BY-4.0) on the LiteRT CompiledModel GPU — fully GPU
 * (108/108 LITERT_CL, ~1.2 ms on a Pixel 8a). CMN'd kaldi fbank [1, 500, 80] -> embedding [256],
 * L2-normalized here.
 */
class SpeakerEmbedder(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val FRAMES = 500
        const val SAMPLES = 400 + (FRAMES - 1) * 160    // 80240 = 5.015 s @ 16 kHz
        const val DIM = 256
    }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "wespeaker_emb_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** fbank: [500][80] CMN'd. Returns the L2-normalized 256-d embedding. */
    fun embed(fbank: Array<FloatArray>): FloatArray {
        require(fbank.size == FRAMES)
        val flat = FloatArray(FRAMES * 80)
        for (t in 0 until FRAMES) System.arraycopy(fbank[t], 0, flat, t * 80, 80)
        inBuf[0].writeFloat(flat)
        model.run(inBuf, outBuf)
        val e = outBuf[0].readFloat()
        var n = 0f
        for (v in e) n += v * v
        n = sqrt(n) + 1e-9f
        for (i in e.indices) e[i] /= n
        return e
    }

    override fun close() {
        inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close()
    }
}
