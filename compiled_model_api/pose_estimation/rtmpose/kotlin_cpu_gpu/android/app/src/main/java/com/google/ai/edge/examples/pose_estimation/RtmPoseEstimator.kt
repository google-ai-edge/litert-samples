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


package com.google.ai.edge.examples.pose_estimation
import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * RTMPose-s (mmpose, CSPNeXt + RTMCC/SimCC head) 2D human pose on the LiteRT CompiledModel GPU.
 *   image[1,3,256,192] (ImageNet 0-255 normalized) -> simcc_x[1,17,384], simcc_y[1,17,512]
 *
 * Top-down: expects a single person, roughly centered, in a 192x256 (WxH) crop. SimCC decodes each of the
 * 17 COCO keypoints by argmax over its 1D x/y bins (bins = pixels x split=2). ~4 ms on a Pixel 8a, fully GPU.
 *
 * Two GPU re-authorings (both numerically exact, baked into the .tflite):
 *  - RTMCC ScaleNorm (RMS): its input reaches ~|274|, so sum(x^2) overflows fp16 (65504) on the Mali delegate
 *    -> norm=inf -> x/inf=0 (all-zero head). Fixed by scaling x down before squaring (SafeRMSNorm).
 *  - GAU attention act@act BMM -> broadcast-multiply + reduce-sum (Mali mis-computes act@act BMM).
 */
class RtmPoseEstimator(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val W = 192
        const val H = 256
        const val K = 17                 // COCO keypoints
        const val SPLIT = 2              // SimCC bin split factor (bins = pixels * split)
        const val MODEL = "rtmpose_s_fp16.tflite"
        val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    /** x,y in the 192x256 crop; score in [0,1]. */
    data class Keypoint(val x: Float, val y: Float, val score: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: W*H*3 row-major [0,255] (the 192x256 person crop). Returns 17 keypoints. */
    fun estimate(rgb: FloatArray): List<Keypoint> {
        val hw = W * H
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        // Outputs are simcc_x [K*W*SPLIT] and simcc_y [K*H*SPLIT]; pick each by length.
        val a = outBuf[0].readFloat(); val b = outBuf[1].readFloat()
        val xBins = W * SPLIT; val yBins = H * SPLIT
        val sx = if (a.size == K * xBins) a else b
        val sy = if (a.size == K * yBins) a else b
        val out = ArrayList<Keypoint>(K)
        for (k in 0 until K) {
            val (xi, xv) = argmax(sx, k * xBins, xBins)
            val (yi, yv) = argmax(sy, k * yBins, yBins)
            out.add(Keypoint(xi / SPLIT.toFloat(), yi / SPLIT.toFloat(), (xv + yv) / 2f))
        }
        return out
    }

    private fun argmax(a: FloatArray, off: Int, n: Int): Pair<Int, Float> {
        var bi = 0; var bv = a[off]
        for (i in 1 until n) { val v = a[off + i]; if (v > bv) { bv = v; bi = i } }
        return bi to bv
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
