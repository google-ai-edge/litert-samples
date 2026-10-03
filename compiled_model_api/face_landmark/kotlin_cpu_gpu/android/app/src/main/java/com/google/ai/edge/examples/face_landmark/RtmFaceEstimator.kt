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

package com.google.ai.edge.examples.face_landmark

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * RTMPose-m face alignment (mmpose, WFLW) on the LiteRT CompiledModel GPU.
 *   face[1,3,256,256] (mmpose mean/std) -> simcc_x[1,98,512], simcc_y[1,98,512]
 *
 * 98 dense WFLW landmarks (contour, eyebrows, eyes, nose, mouth, pupils), decoded by argmax
 * over each 1D x/y SimCC (bins = pixels × split=2). ~4 ms on a Pixel 8a, fully GPU.
 * output[0]=simcc_x, output[1]=simcc_y.
 */
class RtmFaceEstimator(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val W = 256
        const val H = 256
        const val K = 98                 // WFLW landmarks
        const val SPLIT = 2
        const val MODEL = "rtm_face_fp16.tflite"
        val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    data class Point(val x: Float, val y: Float, val score: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: W*H*3 row-major [0,255] (a centered face crop). Returns 98 landmarks in crop pixels. */
    fun estimate(rgb: FloatArray): List<Point> {
        val hw = W * H
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val sx = outBuf[0].readFloat()   // [K * W * SPLIT]
        val sy = outBuf[1].readFloat()   // [K * H * SPLIT]
        val xBins = W * SPLIT
        val yBins = H * SPLIT
        val out = ArrayList<Point>(K)
        for (k in 0 until K) {
            val (xi, xv) = argmax(sx, k * xBins, xBins)
            val (yi, yv) = argmax(sy, k * yBins, yBins)
            out.add(Point(xi / SPLIT.toFloat(), yi / SPLIT.toFloat(), (xv + yv) / 2f))
        }
        return out
    }

    private fun argmax(a: FloatArray, off: Int, n: Int): Pair<Int, Float> {
        var bi = 0
        var bv = a[off]
        for (i in 1 until n) {
            val v = a[off + i]
            if (v > bv) {
                bv = v
                bi = i
            }
        }
        return bi to bv
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
