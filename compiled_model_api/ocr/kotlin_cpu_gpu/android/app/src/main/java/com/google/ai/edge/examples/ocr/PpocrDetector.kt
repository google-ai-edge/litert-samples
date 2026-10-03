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
 * PP-OCRv5 mobile text detection (DBNet) on the LiteRT CompiledModel GPU.
 *   image[1,3,640,640] -> prob map[1,1,640,640]  (sigmoid text-region probability)
 * The DB head's ConvTranspose2d were rewritten to a GPU-clean ZeroStuffConvT2d (TRANSPOSE_CONV is
 * rejected by Mali), so the whole detector runs on the GPU. Box extraction (threshold + connected
 * components + unclip) runs on CPU here — the standard codec/OCR deployment split.
 */
class PpocrDetector(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 640
        const val MODEL = "ppocr_det_fp16.tflite"
        // ImageNet normalization (PP-OCR det preprocessing)
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        const val THRESH = 0.3f
        const val BOX_THRESH = 0.5f
        const val MIN_SIZE = 6
    }

    data class Box(val x0: Int, val y0: Int, val x1: Int, val y1: Int)

    private fun load(name: String): CompiledModel {
        val f = File(ctx.filesDir, name)
        check(f.exists()) { "Model not found: $name. Push first: scripts/install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }

    private val det = load(MODEL)
    private val inBuf = det.createInputBuffers()
    private val outBuf = det.createOutputBuffers()

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns the [SIZE*SIZE] prob map. */
    fun probMap(rgb: FloatArray): FloatArray {
        val chw = FloatArray(3 * SIZE * SIZE)
        val hw = SIZE * SIZE
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        det.run(inBuf, outBuf)
        return outBuf[0].readFloat()           // [SIZE*SIZE]
    }

    /**
     * Threshold the prob map, find connected text regions, return unclipped
     * boxes (in SIZE space).
     */
    fun boxes(prob: FloatArray): List<Box> {
        val n = SIZE * SIZE
        val bin = BooleanArray(n) { prob[it] > THRESH }
        val label = IntArray(n) { -1 }
        val out = ArrayList<Box>()
        val stack = ArrayDeque<Int>()
        for (start in 0 until n) {
            if (!bin[start] || label[start] != -1) continue
            // flood fill (4-connectivity)
            var x0 = SIZE
            var y0 = SIZE
            var x1 = 0
            var y1 = 0
            var area = 0
            var scoreSum = 0f
            stack.addLast(start)
            label[start] = start
            while (stack.isNotEmpty()) {
                val p = stack.removeLast()
                val px = p % SIZE
                val py = p / SIZE
                if (px < x0) x0 = px
                if (px > x1) x1 = px
                if (py < y0) y0 = py
                if (py > y1) y1 = py
                area++
                scoreSum += prob[p]
                if (px > 0 && bin[p - 1] && label[p - 1] == -1) {
                    label[p - 1] = start
                    stack.addLast(p - 1)
                }
                if (px < SIZE - 1 && bin[p + 1] && label[p + 1] == -1) {
                    label[p + 1] = start
                    stack.addLast(p + 1)
                }
                if (py > 0 && bin[p - SIZE] && label[p - SIZE] == -1) {
                    label[p - SIZE] = start
                    stack.addLast(p - SIZE)
                }
                if (py < SIZE - 1 && bin[p + SIZE] && label[p + SIZE] == -1) {
                    label[p + SIZE] = start
                    stack.addLast(p + SIZE)
                }
            }
            val w = x1 - x0 + 1
            val h = y1 - y0 + 1
            if (w < MIN_SIZE || h < MIN_SIZE) continue
            if (scoreSum / area < BOX_THRESH) continue
            // unclip: expand by ~ a fraction of the box height (approximates DB's unclip dilation)
            val pad = (minOf(w, h) * 0.35f).toInt().coerceIn(2, 24)
            out.add(Box((x0 - pad).coerceAtLeast(0), (y0 - pad).coerceAtLeast(0),
                        (x1 + pad).coerceAtMost(SIZE - 1), (y1 + pad).coerceAtMost(SIZE - 1)))
        }
        // sort top-to-bottom, then left-to-right
        return out.sortedWith(compareBy({ it.y0 / 16 }, { it.x0 }))
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        det.close()
    }
}
