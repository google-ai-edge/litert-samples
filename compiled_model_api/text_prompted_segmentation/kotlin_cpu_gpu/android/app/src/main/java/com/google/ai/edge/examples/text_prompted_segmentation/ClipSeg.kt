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

package com.google.ai.edge.examples.text_prompted_segmentation

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * CLIPSeg text-prompted segmentation on the LiteRT CompiledModel API (three graphs):
 *   text    : token embeddings [1,77,512] -> hidden [1,77,512]      (GPU; host EOT row @ text_proj -> cond[512])
 *   vision  : image [1,3,352,352] -> t3, t6, t9 (each [1,485,768])  (GPU)
 *   decoder : t3, t6, t9, cond[512] -> logits [1,352,352]           (CPU — the decoder's 4-head/head_dim-16
 *             attention fp16-miscomputes on the Mali GPU delegate; it's tiny so CPU is exact + fast)
 * Host: BPE tokenize, token-embedding lookup (f16 table), CLIP normalization, sigmoid mask.
 */
class ClipSeg(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 352
        const val TOK = 77
        const val TDIM = 512
        const val PROJ = 512
        // CLIP normalization
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
    }

    private fun f(name: String) = File(ctx.filesDir, name).also {
        check(it.exists()) { "Missing ${it.name}. Run scripts/install_to_device.sh first." }
    }

    private val tokenizer = BpeTokenizer(ctx)
    private val textModel = CompiledModel.create(f("clipseg_text_fp16.tflite").absolutePath,
        CompiledModel.Options(Accelerator.GPU), null)
    private val textIn = textModel.createInputBuffers()
    private val textOut = textModel.createOutputBuffers()
    private val visModel = CompiledModel.create(f("clipseg_vision_fp16.tflite").absolutePath,
        CompiledModel.Options(Accelerator.GPU), null)
    private val visIn = visModel.createInputBuffers()
    private val visOut = visModel.createOutputBuffers()
    private val decModel = CompiledModel.create(f("clipseg_decoder.tflite").absolutePath,
        CompiledModel.Options(Accelerator.CPU), null)
    private val decIn = decModel.createInputBuffers()
    private val decOut = decModel.createOutputBuffers()

    private val tokEmb = loadF16(f("token_embedding_f16.bin"))   // [vocab*512]
    private val textProj = loadF16(f("text_projection_f16.bin")) // [512*512] (in-major: proj[o] = sum_i h[i]*W[i*512+o])

    private fun loadF16(file: File): FloatArray {
        val bytes = file.readBytes()
        val n = bytes.size / 2
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(n)
        for (i in 0 until n) {
            out[i] = halfToFloat(bb.short.toInt() and 0xFFFF)
        }
        return out
    }

    private fun halfToFloat(h: Int): Float {
        val s = (h ushr 15) and 0x1
        val e = (h ushr 10) and 0x1F
        val m = h and 0x3FF
        val bits = when {
            e == 0 -> if (m == 0) s shl 31 else {
                var mant = m
                var exp = -1
                do {
                    mant = mant shl 1
                    exp++
                } while (mant and 0x400 == 0)
                (s shl 31) or ((127 - 15 - exp) shl 23) or ((mant and 0x3FF) shl 13)
            }
            e == 0x1F -> (s shl 31) or (0xFF shl 23) or (m shl 13)
            else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
        }
        return Float.fromBits(bits)
    }

    /** Encode a prompt to the conditional embedding [512]. */
    private fun conditional(prompt: String): FloatArray {
        val (ids, eot) = tokenizer.encode(prompt)
        val emb = FloatArray(TOK * TDIM)
        for (t in 0 until TOK) {
            System.arraycopy(tokEmb, ids[t] * TDIM, emb, t * TDIM, TDIM)
        }
        textIn[0].writeFloat(emb)
        textModel.run(textIn, textOut)
        val hid = textOut[0].readFloat()                        // [77*512]
        val eotRow = FloatArray(TDIM)
        System.arraycopy(hid, eot * TDIM, eotRow, 0, TDIM)
        val cond = FloatArray(PROJ)
        for (o in 0 until PROJ) {
            var acc = 0f
            for (i in 0 until TDIM) {
                acc += eotRow[i] * textProj[i * PROJ + o]
            }
            cond[o] = acc
        }
        return cond
    }

    private fun preprocess(bm: Bitmap): FloatArray {
        val s = Bitmap.createScaledBitmap(bm, SIZE, SIZE, true)
        val px = IntArray(SIZE * SIZE)
        s.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        val out = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in px.indices) {
            val p = px[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            out[i] = (r - MEAN[0]) / STD[0]
            out[plane + i] = (g - MEAN[1]) / STD[1]
            out[2 * plane + i] = (b - MEAN[2]) / STD[2]
        }
        return out
    }

    /**
     * Run the vision graph once per image; cache t3/t6/t9 so re-prompting the same image only
     * re-runs the (tiny) text + decoder graphs.
     */
    private var cachedKey: Bitmap? = null
    private lateinit var t3: FloatArray
    private lateinit var t6: FloatArray
    private lateinit var t9: FloatArray

    private fun runVision(bm: Bitmap) {
        if (cachedKey === bm) return
        visIn[0].writeFloat(preprocess(bm))
        visModel.run(visIn, visOut)
        t3 = visOut[0].readFloat()
        t6 = visOut[1].readFloat()
        t9 = visOut[2].readFloat()
        cachedKey = bm
    }

    /** @return sigmoid mask [SIZE*SIZE] in [0,1] for the prompt over the image. */
    fun segment(bm: Bitmap, prompt: String): FloatArray {
        runVision(bm)
        val cond = conditional(prompt)
        decIn[0].writeFloat(t3)
        decIn[1].writeFloat(t6)
        decIn[2].writeFloat(t9)
        decIn[3].writeFloat(cond)
        decModel.run(decIn, decOut)
        val logits = decOut[0].readFloat()                      // [352*352]
        val mask = FloatArray(logits.size)
        for (i in logits.indices) {
            mask[i] = 1f / (1f + kotlin.math.exp(-logits[i]))
        }
        return mask
    }

    override fun close() {
        textIn.forEach { it.close() }
        textOut.forEach { it.close() }
        textModel.close()
        visIn.forEach { it.close() }
        visOut.forEach { it.close() }
        visModel.close()
        decIn.forEach { it.close() }
        decOut.forEach { it.close() }
        decModel.close()
    }
}
