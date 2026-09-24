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

package com.google.ai.edge.examples.dinov2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.sqrt

/**
 * DINOv2 ViT-S/14 dense feature extraction on the LiteRT CompiledModel GPU, plus
 * a host-side PCA of the patch tokens into an RGB "what the backbone sees" map.
 *
 * The model runs at a fixed 448×448 (32×32 = 1024 patch tokens) fully on the GPU
 * delegate. The only host work is a top-3 PCA of the 1024×384 token matrix, whose
 * three components are mapped to R, G, B per patch. Semantically similar patches
 * (object parts vs background) land near each other in feature space and so share
 * a color.
 *
 * Input : [1, 3, 448, 448] NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1024, 384] patch tokens.
 */
class Dinov2Features(context: Context) : AutoCloseable {

    companion object {
        const val SIZE = 448
        const val GRID = 32          // 448 / 14
        const val N_PATCH = GRID * GRID
        const val DIM = 384
        private const val PCA_ITERS = 40
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val model = CompiledModel.create(
        context.assets, "dinov2_s_fp16.tflite",
        CompiledModel.Options(Accelerator.GPU), null)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()
    private val input = FloatArray(3 * SIZE * SIZE)

    /** Runs DINOv2 and returns the 32×32 PCA feature map as an ARGB bitmap. */
    fun featureMap(bitmap: Bitmap): Bitmap {
        preprocess(bitmap)
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return pcaToBitmap(outputs[0].readFloat())   // [1024*384]
    }

    /** Resize to 448 -> ImageNet-normalized NCHW into [input]. */
    private fun preprocess(bitmap: Bitmap) {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        val pixels = IntArray(SIZE * SIZE)
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            input[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            input[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
    }

    /**
     * Top-3 PCA of the [N_PATCH, DIM] token matrix -> a GRID×GRID ARGB bitmap.
     * Works on the DIM×DIM covariance with power iteration + deflation, which is
     * far smaller than the token Gram matrix.
     */
    private fun pcaToBitmap(feats: FloatArray): Bitmap {
        val centered = meanCenter(feats)
        val cov = covariance(centered)
        val proj = Array(3) { FloatArray(N_PATCH) }
        for (c in 0 until 3) {
            val v = topEigenvector(cov)
            for (i in 0 until N_PATCH) {
                var s = 0f
                val row = i * DIM
                for (d in 0 until DIM) {
                    s += centered[row + d] * v[d]
                }
                proj[c][i] = s
            }
            deflate(cov, v)
        }
        return renderRgb(proj)
    }

    /** Subtracts the per-channel mean from the [N_PATCH, DIM] matrix. */
    private fun meanCenter(feats: FloatArray): FloatArray {
        val mean = FloatArray(DIM)
        for (i in 0 until N_PATCH) {
            val row = i * DIM
            for (d in 0 until DIM) {
                mean[d] += feats[row + d]
            }
        }
        for (d in 0 until DIM) {
            mean[d] /= N_PATCH
        }
        val out = FloatArray(feats.size)
        for (i in 0 until N_PATCH) {
            val row = i * DIM
            for (d in 0 until DIM) {
                out[row + d] = feats[row + d] - mean[d]
            }
        }
        return out
    }

    /** DIM×DIM covariance Xᵀ·X (row-major) of a centered [N_PATCH, DIM] matrix. */
    private fun covariance(x: FloatArray): FloatArray {
        val cov = FloatArray(DIM * DIM)
        for (i in 0 until N_PATCH) {
            val row = i * DIM
            for (a in 0 until DIM) {
                val xa = x[row + a]
                if (xa == 0f) {
                    continue
                }
                val base = a * DIM
                for (b in a until DIM) {
                    cov[base + b] += xa * x[row + b]
                }
            }
        }
        for (a in 0 until DIM) {
            for (b in a + 1 until DIM) {
                cov[b * DIM + a] = cov[a * DIM + b]
            }
        }
        return cov
    }

    /** Leading eigenvector of [cov] by power iteration. */
    private fun topEigenvector(cov: FloatArray): FloatArray {
        var v = FloatArray(DIM) { if (it % 2 == 0) 1f else -1f }
        v = normalize(v)
        repeat(PCA_ITERS) {
            val next = FloatArray(DIM)
            for (a in 0 until DIM) {
                var s = 0f
                val base = a * DIM
                for (b in 0 until DIM) {
                    s += cov[base + b] * v[b]
                }
                next[a] = s
            }
            v = normalize(next)
        }
        return v
    }

    /** Removes the rank-1 component λvvᵀ from [cov] in place. */
    private fun deflate(cov: FloatArray, v: FloatArray) {
        val cv = FloatArray(DIM)
        for (a in 0 until DIM) {
            var s = 0f
            val base = a * DIM
            for (b in 0 until DIM) {
                s += cov[base + b] * v[b]
            }
            cv[a] = s
        }
        var lambda = 0f
        for (a in 0 until DIM) {
            lambda += v[a] * cv[a]
        }
        for (a in 0 until DIM) {
            val base = a * DIM
            for (b in 0 until DIM) {
                cov[base + b] -= lambda * v[a] * v[b]
            }
        }
    }

    private fun normalize(v: FloatArray): FloatArray {
        var n = 0f
        for (x in v) {
            n += x * x
        }
        n = sqrt(n) + 1e-12f
        return FloatArray(v.size) { v[it] / n }
    }

    /** Min-max normalize the 3 projections and pack into a GRID×GRID bitmap. */
    private fun renderRgb(proj: Array<FloatArray>): Bitmap {
        val channel = Array(3) { c ->
            var lo = proj[c][0]
            var hi = proj[c][0]
            for (x in proj[c]) {
                if (x < lo) {
                    lo = x
                }
                if (x > hi) {
                    hi = x
                }
            }
            val span = (hi - lo).takeIf { it > 1e-6f } ?: 1f
            IntArray(N_PATCH) { (((proj[c][it] - lo) / span) * 255f).toInt().coerceIn(0, 255) }
        }
        val pixels = IntArray(N_PATCH)
        for (i in 0 until N_PATCH) {
            pixels[i] = Color.rgb(channel[0][i], channel[1][i], channel[2][i])
        }
        return Bitmap.createBitmap(pixels, GRID, GRID, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        model.close()
    }
}
