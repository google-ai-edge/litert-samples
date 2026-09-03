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

package com.google.ai.edge.examples.low_light_enhancement

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * CPGA-Net (Shyandram, IJPRAI) low-light image enhancement on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (RGB [0,1]) -> enhanced[1,3,256,256] ([0,1])
 *
 * An extremely small (0.025 M params / 0.1 MB fp16 — the smallest in
 * this repo) channel-prior + gamma-correction CNN. ~2 ms / 256×256 on a
 * Pixel 8a, fully GPU. The gamma correction `x^γ` is converted to
 * `exp(γ·log x)`.
 */
class LowLightEnhancer(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object { const val SIZE = 256 }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "cpga_fp16.tflite")
        check(f.exists()) {
            "Model not found: cpga_fp16.tflite. " +
                "Push first: scripts/install_to_device.sh"
        }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns the enhanced SIZE×SIZE bitmap. */
    fun enhance(rgb: FloatArray): Bitmap {
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {                          // RGB [0,255] -> [0,1], NCHW planar
            chw[i] = rgb[i * 3] / 255f
            chw[hw + i] = rgb[i * 3 + 1] / 255f
            chw[2 * hw + i] = rgb[i * 3 + 2] / 255f
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val out = outBuf[0].readFloat()                  // [3*hw], [0,1]
        val px = IntArray(hw)
        for (i in 0 until hw) {
            val r = (out[i] * 255f).coerceIn(0f, 255f).toInt()
            val g = (out[hw + i] * 255f).coerceIn(0f, 255f).toInt()
            val b = (out[2 * hw + i] * 255f).coerceIn(0f, 255f).toInt()
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
