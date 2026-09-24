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

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * pyannote segmentation-3.0 (PyanNet: SincNet + BiLSTM, MIT) via onnxruntime CPU — the LSTM has
 * no Mali GPU kernel, but the model is tiny (1.5 M params, ~5.9 MB) and fast on CPU.
 * 10 s window [1,1,160000] -> powerset log-probs [1, 589, 7].
 */
class SegmentationOnnx(ctx: Context) : Closeable {

    companion object {
        const val SR = 16000
        const val WINDOW = 10 * SR
        const val MAX_LOCAL_SPEAKERS = 3
        // powerset classes: ∅, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
        val POWERSET = arrayOf(intArrayOf(), intArrayOf(0), intArrayOf(1), intArrayOf(2),
                               intArrayOf(0, 1), intArrayOf(0, 2), intArrayOf(1, 2))
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = run {
        val f = File(ctx.filesDir, "pyannote_seg30.onnx")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        env.createSession(f.absolutePath, OrtSession.SessionOptions())
    }
    var frames = 0
        private set

    /** @return per-frame binary activity [frames][3] for one 10 s window. */
    fun run(window: FloatArray): Array<FloatArray> {
        require(window.size == WINDOW)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(window), longArrayOf(1, 1, WINDOW.toLong()))
            .use { input ->
                session.run(mapOf(session.inputNames.first() to input)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    val ps = (out[0].value as Array<Array<FloatArray>>)[0]  // [frames][7]
                    frames = ps.size
                    val act = Array(ps.size) { FloatArray(MAX_LOCAL_SPEAKERS) }
                    for (t in ps.indices) {
                        var best = 0
                        for (c in 1 until 7) {
                            if (ps[t][c] > ps[t][best]) {
                                best = c
                            }
                        }
                        for (s in POWERSET[best]) {
                            act[t][s] = 1f
                        }
                    }
                    return act
                }
            }
    }

    override fun close() {
        session.close()
    }
}
