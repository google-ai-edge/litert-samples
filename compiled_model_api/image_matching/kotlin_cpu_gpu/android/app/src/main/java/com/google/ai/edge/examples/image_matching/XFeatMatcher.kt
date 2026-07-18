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

package com.google.ai.edge.examples.image_matching

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * XFeat (CVPR 2024, Apache-2.0) local feature extraction on the LiteRT CompiledModel GPU —
 * fully GPU (72/72 LITERT_CL, ~0.4 ms/frame on a Pixel 8a). Host: per-image normalization,
 * keypoint decode (8x8-cell logits + dustbin), descriptor bilinear sampling, and
 * mutual-nearest-neighbor matching.
 */
class XFeatMatcher(ctx: Context) : Closeable {

    companion object {
        const val W = 640
        const val H = 480
        const val GW = 80          // W/8
        const val GH = 60          // H/8
        const val D = 64           // descriptor dim
        const val TOP_K = 800
        const val MIN_COSSIM = 0.82f
    }

    data class Features(val xs: FloatArray, val ys: FloatArray, val scores: FloatArray,
                        val desc: Array<FloatArray>)
    data class Match(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val sim: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "xfeat_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** bitmap (any size) -> grayscale 640x480, per-image instance norm (host side). */
    fun preprocess(bm: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bm, W, H, true)
        val px = IntArray(W * H)
        scaled.getPixels(px, 0, W, 0, 0, W, H)
        val g = FloatArray(W * H)
        var mean = 0.0
        for (i in px.indices) {
            val p = px[i]
            g[i] = (0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) +
                    0.114f * (p and 0xFF))
            mean += g[i]
        }
        mean /= g.size
        var varSum = 0.0
        for (v in g) {
            val d = v - mean
            varSum += d * d
        }
        val inv = (1.0 / sqrt(varSum / g.size + 1e-5)).toFloat()
        for (i in g.indices) {
            g[i] = ((g[i] - mean.toFloat()) * inv)
        }
        return g
    }

    /** Run the GPU graph and decode top-K keypoints + descriptors. */
    fun extract(gray: FloatArray): Features {
        inBuf[0].writeFloat(gray)
        model.run(inBuf, outBuf)
        // Output buffers follow the model's signature order: feats, keypoint logits, heatmap.
        val feats = outBuf[0].readFloat()      // [64, 60, 80]
        val klog = outBuf[1].readFloat()       // [65, 60, 80] cell logits (64 pos + dustbin)
        val heat = outBuf[2].readFloat()       // [60, 80] reliability

        // per-cell softmax over 65 -> pixel scores * reliability
        val cand = ArrayList<Triple<Int, Int, Float>>(GW * GH)
        val cell = FloatArray(65)
        val scoreMap = FloatArray(W * H)
        for (cy in 0 until GH) {
            for (cx in 0 until GW) {
                var mx = Float.NEGATIVE_INFINITY
                for (c in 0 until 65) {
                    cell[c] = klog[(c * GH + cy) * GW + cx]
                    if (cell[c] > mx) {
                        mx = cell[c]
                    }
                }
                var sum = 0f
                for (c in 0 until 65) {
                    cell[c] = exp(cell[c] - mx)
                    sum += cell[c]
                }
                val rel = heat[cy * GW + cx]
                for (c in 0 until 64) {
                    val py = cy * 8 + c / 8
                    val pxx = cx * 8 + c % 8
                    scoreMap[py * W + pxx] = (cell[c] / sum) * rel
                }
            }
        }
        // 5x5 NMS + top-K
        for (y in 0 until H) {
            for (x in 0 until W) {
                val s = scoreMap[y * W + x]
                if (s < 1e-4f) continue
                var isMax = true
                loop@ for (dy in -2..2) for (dx in -2..2) {
                    val yy = y + dy
                    val xx = x + dx
                    if (yy in 0 until H && xx in 0 until W && scoreMap[yy * W + xx] > s) {
                        isMax = false
                        break@loop
                    }
                }
                if (isMax) {
                    cand.add(Triple(x, y, s))
                }
            }
        }
        cand.sortByDescending { it.third }
        val n = minOf(TOP_K, cand.size)
        val xs = FloatArray(n)
        val ys = FloatArray(n)
        val sc = FloatArray(n)
        val desc = Array(n) { FloatArray(D) }
        for (i in 0 until n) {
            val (x, y, s) = cand[i]
            xs[i] = x.toFloat()
            ys[i] = y.toFloat()
            sc[i] = s
            // bilinear sample the [64, 60, 80] feature map at (x/8, y/8), then L2 normalize
            val fx = (x / 8f - 0.5f).coerceIn(0f, GW - 1.001f)
            val fy = (y / 8f - 0.5f).coerceIn(0f, GH - 1.001f)
            val x0 = fx.toInt()
            val y0 = fy.toInt()
            val ax = fx - x0
            val ay = fy - y0
            var norm = 0f
            for (d in 0 until D) {
                val base = d * GH * GW
                val v = (feats[base + y0 * GW + x0] * (1 - ax) * (1 - ay)
                        + feats[base + y0 * GW + x0 + 1] * ax * (1 - ay)
                        + feats[base + (y0 + 1) * GW + x0] * (1 - ax) * ay
                        + feats[base + (y0 + 1) * GW + x0 + 1] * ax * ay)
                desc[i][d] = v
                norm += v * v
            }
            norm = sqrt(norm) + 1e-9f
            for (d in 0 until D) {
                desc[i][d] /= norm
            }
        }
        return Features(xs, ys, sc, desc)
    }

    /** Mutual-nearest-neighbor matching with a cosine-similarity floor. */
    fun match(a: Features, b: Features): List<Match> {
        val na = a.desc.size
        val nb = b.desc.size
        if (na == 0 || nb == 0) return emptyList()
        val bestB = IntArray(na) { -1 }
        val simB = FloatArray(na) { -2f }
        val bestA = IntArray(nb) { -1 }
        val simA = FloatArray(nb) { -2f }
        for (i in 0 until na) {
            for (j in 0 until nb) {
                var s = 0f
                val da = a.desc[i]
                val db = b.desc[j]
                for (d in 0 until D) {
                    s += da[d] * db[d]
                }
                if (s > simB[i]) {
                    simB[i] = s
                    bestB[i] = j
                }
                if (s > simA[j]) {
                    simA[j] = s
                    bestA[j] = i
                }
            }
        }
        val out = ArrayList<Match>()
        for (i in 0 until na) {
            val j = bestB[i]
            if (j >= 0 && bestA[j] == i && simB[i] >= MIN_COSSIM)
                out.add(Match(a.xs[i], a.ys[i], b.xs[j], b.ys[j], simB[i]))
        }
        return out
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
