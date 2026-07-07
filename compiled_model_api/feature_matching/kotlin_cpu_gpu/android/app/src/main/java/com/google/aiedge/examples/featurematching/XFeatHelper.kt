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

package com.google.aiedge.examples.featurematching

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * XFeat (Accelerated Features, Apache-2.0) lightweight local-feature extractor on LiteRT
 * CompiledModel (CPU / GPU). The pure-CNN net (re-authored GPU-clean via litert_torch: host gray +
 * InstanceNorm, `_unfold2d` → one-hot space-to-depth conv) is full LITERT_CL resident on the Pixel 8a
 * (~0.4 ms).
 *
 *   input : images [1, 480, 640, 1] NHWC, grayscale, host InstanceNorm
 *   output: feats[1,64,60,80] (descriptors) + keypoints[1,65,60,80] (logits) + heatmap[1,1,60,80]
 *
 * Keypoint decode (per 8×8 cell), descriptor sampling/L2-norm, and mutual-NN matching run here.
 */
class XFeatHelper(
    private val context: Context,
    private var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
) {
    companion object {
        private const val TAG = "XFeat"
        val DEFAULT_DELEGATE = AcceleratorEnum.GPU
        const val IN_W = 640
        const val IN_H = 480
        const val GW = IN_W / 8   // 80
        const val GH = IN_H / 8   // 60
        const val DESC = 64
        const val TOP_K = 1024
        const val MIN_COSSIM = 0.82f
        const val MODEL_FILE = "xfeat.tflite"

        fun toAccelerator(a: AcceleratorEnum) =
            if (a == AcceleratorEnum.CPU) Accelerator.CPU else Accelerator.GPU
    }

    enum class AcceleratorEnum { CPU, GPU }

    /** A keypoint with its L2-normalized descriptor, in ORIGINAL-image pixel coords. */
    class Feature(val x: Float, val y: Float, val score: Float, val desc: FloatArray)

    private var model: CompiledModel? = null
    private val dispatcher = Dispatchers.IO.limitedParallelism(1, "XFeatModel")
    private val inputFloats = FloatArray(IN_W * IN_H) // 1 channel
    private val pixels = IntArray(IN_W * IN_H)

    suspend fun init() {
        cleanup()
        withContext(dispatcher) {
            model = CompiledModel.create(
                context.assets, MODEL_FILE, CompiledModel.Options(toAccelerator(delegate)), null
            )
            Log.i(TAG, "Created CompiledModel $MODEL_FILE on $delegate")
        }
    }

    suspend fun cleanup() = withContext(dispatcher) {
        model?.close()
        model = null
    }

    fun setDelegate(d: AcceleratorEnum) { delegate = d }

    /** Extract up to TOP_K features from [bitmap] (resized to 640×480, scores in original coords). */
    suspend fun extract(bitmap: Bitmap): List<Feature> = withContext(dispatcher) {
        val m = model ?: return@withContext emptyList()
        val scaled = Bitmap.createScaledBitmap(bitmap, IN_W, IN_H, true)
        scaled.getPixels(pixels, 0, IN_W, 0, 0, IN_W, IN_H)
        // grayscale (luma) + per-image InstanceNorm (host)
        var mean = 0.0
        for (i in pixels.indices) {
            val p = pixels[i]
            val g = (0.299f * Color.red(p) + 0.587f * Color.green(p) + 0.114f * Color.blue(p)) / 255f
            inputFloats[i] = g
            mean += g
        }
        val mu = (mean / pixels.size).toFloat()
        var v = 0.0
        for (g in inputFloats) {
            v += (g - mu) * (g - mu)
        }
        val inv = (1.0 / sqrt(v / pixels.size + 1e-5)).toFloat()
        for (i in inputFloats.indices) {
            inputFloats[i] = (inputFloats[i] - mu) * inv
        }

        val inBuf = m.createInputBuffers()
        val outBuf = m.createOutputBuffers()
        inBuf[0].writeFloat(inputFloats)
        m.run(inBuf, outBuf)
        // identify outputs by length: 64*GH*GW feats, 65*GH*GW keypoints, GH*GW heatmap
        var feats = FloatArray(0)
        var kpts = FloatArray(0)
        var heat = FloatArray(0)
        for (b in outBuf) {
            val a = b.readFloat()
            when (a.size) {
                DESC * GH * GW -> feats = a
                65 * GH * GW -> kpts = a
                GH * GW -> heat = a
            }
        }
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }

        val sx = bitmap.width.toFloat() / IN_W
        val sy = bitmap.height.toFloat() / IN_H
        decode(feats, kpts, heat, sx, sy)
    }

    /** Per-cell keypoint decode: softmax 65 logits, best of the 64 sub-cells, score×reliability. */
    private fun decode(feats: FloatArray, kpts: FloatArray, heat: FloatArray, sx: Float, sy: Float): List<Feature> {
        val cell = GH * GW
        val out = ArrayList<Feature>(cell)
        val logit = FloatArray(65)
        for (cy in 0 until GH) for (cx in 0 until GW) {
            val o = cy * GW + cx
            var mx = Float.NEGATIVE_INFINITY
            for (c in 0 until 65) {
                val z = kpts[c * cell + o]
                logit[c] = z
                if (z > mx) {
                    mx = z
                }
            }
            var sum = 0f
            for (c in 0 until 65) {
                val e = exp((logit[c] - mx).toDouble()).toFloat()
                logit[c] = e
                sum += e
            }
            // best non-dustbin sub-cell (channels 0..63 = i*8+j)
            var best = 0
            var bestP = -1f
            for (c in 0 until 64) {
                val p = logit[c] / sum
                if (p > bestP) {
                    bestP = p
                    best = c
                }
            }
            val score = bestP * heat[o]   // reliability-weighted
            if (score <= 0f) continue
            val i = best / 8
            val j = best % 8
            val px = (cx * 8 + j) * sx
            val py = (cy * 8 + i) * sy
            val d = FloatArray(DESC)
            var n = 0f
            for (c in 0 until DESC) {
                val z = feats[c * cell + o]
                d[c] = z
                n += z * z
            }
            n = (1.0 / sqrt(n + 1e-12)).toFloat()
            for (c in 0 until DESC) {
                d[c] *= n
            }
            out.add(Feature(px, py, score, d))
        }
        out.sortByDescending { it.score }
        return if (out.size > TOP_K) out.subList(0, TOP_K) else out
    }

    /** Mutual-nearest-neighbour cosine matching (descriptors are L2-normalized → dot = cosine). */
    fun match(a: List<Feature>, b: List<Feature>, minCos: Float = MIN_COSSIM): List<Pair<Int, Int>> {
        if (a.isEmpty() || b.isEmpty()) return emptyList()
        val a2b = IntArray(a.size)
        val a2bScore = FloatArray(a.size)
        val b2a = IntArray(b.size) { -1 }
        val b2aScore = FloatArray(b.size) { -1f }
        for (i in a.indices) {
            var bi = -1
            var bs = -1f
            val da = a[i].desc
            for (k in b.indices) {
                var s = 0f
                val db = b[k].desc
                for (c in 0 until DESC) {
                    s += da[c] * db[c]
                }
                if (s > bs) {
                    bs = s
                    bi = k
                }
                if (s > b2aScore[k]) {
                    b2aScore[k] = s
                    b2a[k] = i
                }
            }
            a2b[i] = bi
            a2bScore[i] = bs
        }
        val res = ArrayList<Pair<Int, Int>>()
        for (i in a.indices) {
            val k = a2b[i]
            if (k >= 0 && b2a[k] == i && a2bScore[i] > minCos) {
                res.add(i to k)
            }
        }
        return res
    }
}
