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

package com.google.ai.edge.examples.gaze_estimation

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * L2CS-Net gaze estimation (ResNet50, Gaze360) on the LiteRT CompiledModel GPU.
 *   face[1,3,448,448] (ImageNet-normalized) -> yaw[1,90], pitch[1,90] (softmax over angle bins, baked in)
 *
 * Predicts where a (centered) face is looking. The 90 bins span [-180,180]° at 4° each; the gaze angle is the
 * softmax expectation: `deg = Σ p_i·i · 4 − 180`. ~3 ms / 448x448 on a Pixel 8a, fully GPU. The ResNet stem
 * `MaxPool2d(3,s2,p1)` (which pads with -inf -> a `PADV2` the Mali delegate won't delegate) is re-authored to a
 * zero-pad + valid max-pool (exact since it follows a ReLU); the global pool -> mean(3).mean(2).
 */
class GazeEstimator(ctx: Context) : Closeable {

    companion object {
        const val SIZE = 448
        const val BINS = 90
        const val MODEL = "gaze_fp16.tflite"
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    /** yaw/pitch in degrees (Gaze360 convention). */
    data class Gaze(val yawDeg: Float, val pitchDeg: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: SIZE*SIZE*3 row-major [0,255] (a centered face crop). Returns the gaze angles. */
    fun estimate(rgb: FloatArray): Gaze {
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val yaw = outBuf[0].readFloat()     // softmax [90]
        val pitch = outBuf[1].readFloat()   // softmax [90]
        return Gaze(expect(yaw), expect(pitch))
    }

    private fun expect(p: FloatArray): Float {
        var e = 0f
        for (i in 0 until BINS) e += p[i] * i
        return e * 4f - 180f
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
