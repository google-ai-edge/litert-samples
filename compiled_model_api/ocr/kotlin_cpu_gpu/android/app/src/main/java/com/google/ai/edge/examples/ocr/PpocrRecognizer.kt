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

package com.google.ai.edge.examples.ocr

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * PP-OCRv5 mobile text recognition (PPLCNetV3 + SVTR + CTC) on the LiteRT CompiledModel GPU.
 *   line[1,3,48,320] -> logits[1,T,18385]  (CTC over the char set)
 * SVTR attention's fused-QKV 5D reshape was split into 4D for the GPU; the head is CTC (no
 * autoregressive decoder), so the whole recognizer runs on the GPU. CTC greedy decode is on CPU.
 */
class PpocrRecognizer(private val ctx: Context) : Closeable {

    companion object {
        const val H = 48
        const val W = 320
        const val MODEL = "ppocr_rec_fp16.tflite"
        const val DICT = "ppocrv5_dict.txt"
    }

    private fun load(name: String): CompiledModel {
        val f = File(ctx.filesDir, name)
        check(f.exists()) { "Model not found: $name. Push first: scripts/install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }

    private val rec = load(MODEL)
    private val inBuf = rec.createInputBuffers()
    private val outBuf = rec.createOutputBuffers()
    private val numClasses: Int
    private val chars: Array<String>   // CTC layout: index 0 = blank, then dict, then space

    init {
        val dict = ctx.assets.open(DICT).bufferedReader().readLines()
        chars = (listOf("") + dict + listOf(" ")).toTypedArray()   // blank + dict + space
        numClasses = chars.size                                     // 18385
    }

    /** crop: H*W*3 row-major [0,255] of a resized+padded text line. Returns recognized text. */
    fun recognize(crop: FloatArray): String {
        val chw = FloatArray(3 * H * W)
        val hw = H * W
        for (i in 0 until hw) {                                     // (img/255 - 0.5)/0.5
            chw[i] = (crop[i * 3] / 255f - 0.5f) / 0.5f
            chw[hw + i] = (crop[i * 3 + 1] / 255f - 0.5f) / 0.5f
            chw[2 * hw + i] = (crop[i * 3 + 2] / 255f - 0.5f) / 0.5f
        }
        inBuf[0].writeFloat(chw)
        rec.run(inBuf, outBuf)
        val logits = outBuf[0].readFloat()                         // [T*numClasses]
        val T = logits.size / numClasses
        val sb = StringBuilder()
        var prev = -1
        for (t in 0 until T) {
            var best = 0; var bestV = logits[t * numClasses]
            for (c in 1 until numClasses) {
                val v = logits[t * numClasses + c]
                if (v > bestV) { bestV = v; best = c }
            }
            if (best != prev && best != 0) sb.append(chars[best])  // CTC collapse: drop repeats + blank
            prev = best
        }
        return sb.toString()
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; rec.close() }
}
