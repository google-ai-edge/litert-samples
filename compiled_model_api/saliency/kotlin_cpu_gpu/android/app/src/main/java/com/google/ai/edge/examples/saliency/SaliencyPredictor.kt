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

package com.google.ai.edge.examples.saliency

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * UniSal (rdroste/unisal) visual-saliency prediction on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (ImageNet mean/std) -> saliency[1,1,256,256] (where humans look)
 *
 * MobileNetV2 encoder + bilinear-upsample decoder + a fixed 41×41 Gaussian smoothing, with the
 * SALICON domain pinned. ~3 ms on a Pixel 8a, fully GPU. Output is a single-channel saliency
 * map (higher = more attended); normalize for display.
 */
class SaliencyPredictor(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val SIZE = 256
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "unisal_fp16.tflite")
        check(f.exists()) {
            "Model not found: unisal_fp16.tflite. Push first: scripts/install_to_device.sh"
        }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /**
     * rgb: SIZE*SIZE*3 row-major [0,255]. Returns the SIZE*SIZE saliency map, min-max
     * normalized to [0,1].
     */
    fun predict(rgb: FloatArray): FloatArray {
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val sal = outBuf[0].readFloat()            // [hw]
        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (v in sal) {
            val r = if (v > 0f) v else 0f
            if (r < mn) {
                mn = r
            }
            if (r > mx) {
                mx = r
            }
        }
        val range = (mx - mn).coerceAtLeast(1e-6f)
        val out = FloatArray(hw)
        for (i in 0 until hw) out[i] = ((if (sal[i] > 0f) sal[i] else 0f) - mn) / range
        return out
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
