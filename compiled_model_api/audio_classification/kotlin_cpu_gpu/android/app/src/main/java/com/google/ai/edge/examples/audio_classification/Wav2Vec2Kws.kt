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

package com.google.ai.edge.examples.audio_classification

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * wav2vec2 keyword spotting (superb/wav2vec2-base-superb-ks), on-device, fully on the LiteRT
 * CompiledModel GPU (ML Drift). 12 Speech-Commands labels.
 *
 *   waveform[16000] -[GPU frontend]-> feat[1,49,768] -[GPU head]-> logits[12] -> argmax
 *
 * The model is shipped as TWO GPU graphs because the full 1008-node graph exceeds the Mali
 * shader-compile limit (it fails to compile fused, while each half compiles: frontend 134/134 +
 * head 893/893 LITERT_CL). Both halves run on the GPU; there is no FFT anywhere (raw 16 kHz
 * waveform straight into the 1D-conv feature extractor — no mel step).
 *
 *   frontend = feature_extractor (7 strided 1D convs + GroupNorm) + feature_projection.
 *   head     = encoder (12 transformer layers, pos-conv) + weighted-layer-sum over all 13
 *              hidden states (this checkpoint uses use_weighted_layer_sum) + projector +
 *              mean-pool + classifier.
 *
 * Both tflites are loaded from filesDir (push via scripts/install_to_device.sh).
 */
class Wav2Vec2Kws(private val context: Context) : Closeable {

    companion object {
        const val SAMPLES = 16000   // 1 s @ 16 kHz
        const val FRONTEND = "w2v2_frontend_fp16.tflite"
        const val HEAD = "w2v2_head_fp16.tflite"
        val LABELS = arrayOf(
            "yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go", "_unknown_", "_silence_",
        )
    }

    private fun load(name: String): CompiledModel {
        val f = File(context.filesDir, name)
        check(f.exists()) {
            "Model not found: $name. Push it first:\n  scripts/install_to_device.sh"
        }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }

    private val frontend = load(FRONTEND)
    private val head = load(HEAD)
    private val feIn = frontend.createInputBuffers()
    private val feOut = frontend.createOutputBuffers()
    private val hdIn = head.createInputBuffers()
    private val hdOut = head.createOutputBuffers()

    data class Result(val label: String, val index: Int, val logits: FloatArray, val ms: Long)

    /** Classify a SAMPLES-length 16 kHz mono clip. Pads/truncates to SAMPLES. */
    fun classify(audio: FloatArray): Result {
        val x = FloatArray(SAMPLES)
        System.arraycopy(audio, 0, x, 0, minOf(audio.size, SAMPLES))
        val t0 = System.nanoTime()
        feIn[0].writeFloat(x)
        frontend.run(feIn, feOut)
        val feat = feOut[0].readFloat()        // [1*49*768]
        hdIn[0].writeFloat(feat)
        head.run(hdIn, hdOut)
        val logits = hdOut[0].readFloat()      // [12]
        val ms = (System.nanoTime() - t0) / 1_000_000
        var best = 0
        for (i in logits.indices) {
            if (logits[i] > logits[best]) {
                best = i
            }
        }
        return Result(LABELS[best], best, logits, ms)
    }

    override fun close() {
        feIn.forEach { it.close() }
        feOut.forEach { it.close() }
        hdIn.forEach { it.close() }
        hdOut.forEach { it.close() }
        frontend.close()
        head.close()
    }
}
