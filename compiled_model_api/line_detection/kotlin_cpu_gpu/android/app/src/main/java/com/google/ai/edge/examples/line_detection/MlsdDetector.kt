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


package com.google.ai.edge.examples.line_detection
import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.exp
import kotlin.math.hypot

/**
 * M-LSD-tiny line segment detection (NAVER, MobileNetV2 backbone) on the LiteRT CompiledModel GPU.
 *   image[1,4,512,512] (RGB + ones channel, scaled to [-1,1]) -> tpMap[1,9,256,256]
 *
 * Pure CNN encoder-decoder (bilinear upsample, the only GPU re-authoring being align_corners=True->False).
 * The output is a "TP map": channel 0 = line-center heatmap, channels 1..4 = start/end displacement. The
 * decode (sigmoid + 3x3 NMS + displacement -> endpoints) runs here. ~2 ms / 512x512 on a Pixel 8a, fully GPU.
 */
class MlsdDetector(ctx: Context) : Closeable {

    companion object {
        const val SIZE = 512        // model input
        const val OUT = 256         // tpMap resolution (input/2)
        const val MODEL = "mlsd_fp16.tflite"
        const val FOURTH = (1f / 127.5f) - 1f   // the constant 4th input channel (ones, scaled)
    }

    /** Endpoints in the original image's pixel space. */
    data class Line(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /**
     * rgb: SIZE*SIZE*3 row-major [0,255] (image already resized to 512x512).
     * Returns line segments scaled to [scaleW x scaleH] (the displayed image size).
     */
    fun detect(rgb: FloatArray, scaleW: Float, scaleH: Float, scoreThr: Float = 0.10f, distThr: Float = 20f): List<Line> {
        val hw = SIZE * SIZE
        val chw = FloatArray(4 * hw)
        for (i in 0 until hw) {
            chw[i] = rgb[i * 3] / 127.5f - 1f
            chw[hw + i] = rgb[i * 3 + 1] / 127.5f - 1f
            chw[2 * hw + i] = rgb[i * 3 + 2] / 127.5f - 1f
            chw[3 * hw + i] = FOURTH
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val out = outBuf[0].readFloat()            // [9*256*256] NCHW
        val o = OUT * OUT
        // 3x3 NMS over the sigmoid center map; keep local maxima above threshold.
        val cand = ArrayList<IntArray>()           // (y, x)
        val scores = ArrayList<Float>()
        for (y in 0 until OUT) for (x in 0 until OUT) {
            val c = sigmoid(out[y * OUT + x])
            if (c <= scoreThr) continue
            var isMax = true
            var dy = -1
            loop@ while (dy <= 1) {
                var dx = -1
                while (dx <= 1) {
                    val ny = y + dy; val nx = x + dx
                    if ((dy != 0 || dx != 0) && ny in 0 until OUT && nx in 0 until OUT) {
                        if (sigmoid(out[ny * OUT + nx]) > c) { isMax = false; break@loop }
                    }
                    dx++
                }
                dy++
            }
            if (isMax) { cand.add(intArrayOf(y, x)); scores.add(c) }
        }
        val order = scores.indices.sortedByDescending { scores[it] }.take(200)
        val sx = scaleW / SIZE; val sy = scaleH / SIZE     // tpMap*2 -> 512, then -> display
        val lines = ArrayList<Line>()
        for (idx in order) {
            val (y, x) = cand[idx]
            val dxs = out[1 * o + y * OUT + x]; val dys = out[2 * o + y * OUT + x]
            val dxe = out[3 * o + y * OUT + x]; val dye = out[4 * o + y * OUT + x]
            if (hypot((dxs - dxe).toDouble(), (dys - dye).toDouble()) <= distThr) continue
            lines.add(Line((x + dxs) * 2 * sx, (y + dys) * 2 * sy, (x + dxe) * 2 * sx, (y + dye) * 2 * sy, scores[idx]))
        }
        return lines
    }

    private fun sigmoid(v: Float) = 1f / (1f + exp(-v))

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
