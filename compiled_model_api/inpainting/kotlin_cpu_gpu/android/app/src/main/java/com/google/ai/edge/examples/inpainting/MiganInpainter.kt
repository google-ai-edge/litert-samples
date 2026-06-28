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

package com.google.ai.edge.examples.inpainting

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * MI-GAN (Picsart, ICCV 2023) image inpainting / object removal on the LiteRT CompiledModel GPU.
 *   in[1,4,512,512] = concat(mask-0.5, rgb·mask)  ->  out[1,3,512,512] (the inpainted image, [-1,1])
 *
 * The 4-channel input is `[mask-0.5, masked_rgb]` (mask: 1 = keep, 0 = erase/fill; rgb in [-1,1]). The output
 * is composited back as `rgb·mask + out·(1-mask)` so only the erased region is replaced. A mobile-designed
 * StyleGAN-style generator (separable convs, nearest-upsample, no norm) → ~6 ms / 512x512 on a Pixel 8a,
 * fully GPU.
 */
class MiganInpainter(ctx: Context) : Closeable {

    companion object { const val SIZE = 512 }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "migan_fp16.tflite")
        check(f.exists()) { "Model not found: migan_fp16.tflite. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /**
     * rgb: SIZE*SIZE*3 row-major [0,255]; mask: SIZE*SIZE with 1 = keep, 0 = erase. Returns the inpainted
     * bitmap (erased region filled, the rest left exactly as the input).
     */
    fun inpaint(rgb: FloatArray, mask: FloatArray): Bitmap {
        val hw = SIZE * SIZE
        val chw = FloatArray(4 * hw)
        for (i in 0 until hw) {
            val mk = mask[i]
            chw[i] = mk - 0.5f                                       // ch0: mask - 0.5
            chw[hw + i] = (rgb[i * 3] / 127.5f - 1f) * mk           // ch1: R·mask
            chw[2 * hw + i] = (rgb[i * 3 + 1] / 127.5f - 1f) * mk   // ch2: G·mask
            chw[3 * hw + i] = (rgb[i * 3 + 2] / 127.5f - 1f) * mk   // ch3: B·mask
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val out = outBuf[0].readFloat()                             // [3*hw], [-1,1]
        val px = IntArray(hw)
        for (i in 0 until hw) {
            val mk = mask[i]
            val r = ((rgb[i * 3] / 127.5f - 1f) * mk + out[i] * (1f - mk) + 1f) * 127.5f
            val g = ((rgb[i * 3 + 1] / 127.5f - 1f) * mk + out[hw + i] * (1f - mk) + 1f) * 127.5f
            val b = ((rgb[i * 3 + 2] / 127.5f - 1f) * mk + out[2 * hw + i] * (1f - mk) + 1f) * 127.5f
            px[i] = Color.rgb(r.coerceIn(0f, 255f).toInt(), g.coerceIn(0f, 255f).toInt(), b.coerceIn(0f, 255f).toInt())
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
