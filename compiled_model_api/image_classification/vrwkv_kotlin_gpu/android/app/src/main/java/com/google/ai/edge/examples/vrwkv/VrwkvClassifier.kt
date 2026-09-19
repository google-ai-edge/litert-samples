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

package com.google.ai.edge.examples.vrwkv

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * Vision-RWKV (VRWKV-S) ImageNet-1K classification on the LiteRT CompiledModel GPU.
 *
 * VRWKV is an RWKV-style vision backbone: its token mixer is a bidirectional WKV
 * (a linear-attention scan) rather than softmax self-attention. Because the token
 * count is fixed (14x14 = 196 patches), that WKV is re-authored exactly as a
 * per-channel decay-biased attention and runs fully on the GPU delegate.
 *
 * The graph takes two inputs: the NCHW image and the constant token-distance
 * matrix `dist[t,i] = |t-i|` (fed at runtime so its per-channel decay bias is not
 * baked into the flatbuffer as a large constant). It returns 1000 class logits.
 */
class VrwkvClassifier(context: Context) : AutoCloseable {

    companion object {
        const val SIZE = 224
        private const val GRID = 14
        private const val TOKENS = GRID * GRID          // 196
        private const val RESIZE_SHORT = 256
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    private val model = CompiledModel.create(
        context.assets, "vrwkv_s_fp16.tflite", CompiledModel.Options(Accelerator.GPU), null)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()
    private val labels = context.assets.open("imagenet_classes.txt").bufferedReader()
        .readLines()

    /** Token-distance matrix |t-i|, constant for a fixed 196-token grid. */
    private val dist = FloatArray(TOKENS * TOKENS).also { d ->
        for (t in 0 until TOKENS) {
            for (i in 0 until TOKENS) {
                d[t * TOKENS + i] = kotlin.math.abs(t - i).toFloat()
            }
        }
    }

    private val input = FloatArray(3 * SIZE * SIZE)

    /** One prediction: label plus its softmax probability. */
    data class Prediction(val label: String, val probability: Float)

    /** Classifies [bitmap] and returns the top-[k] predictions. */
    fun classify(bitmap: Bitmap, k: Int = 5): List<Prediction> {
        preprocess(bitmap)
        inputs[0].writeFloat(input)
        inputs[1].writeFloat(dist)
        model.run(inputs, outputs)
        return topK(outputs[0].readFloat(), k)
    }

    /** Resize (short edge 256) -> center-crop 224 -> ImageNet-normalized NCHW. */
    private fun preprocess(bitmap: Bitmap) {
        val w = bitmap.width
        val h = bitmap.height
        val scale = RESIZE_SHORT.toFloat() / minOf(w, h)
        val rw = Math.round(w * scale)
        val rh = Math.round(h * scale)
        val resized = Bitmap.createScaledBitmap(bitmap, rw, rh, true)
        val left = (rw - SIZE) / 2
        val top = (rh - SIZE) / 2
        val pixels = IntArray(SIZE * SIZE)
        resized.getPixels(pixels, 0, SIZE, left, top, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (idx in 0 until plane) {
            val p = pixels[idx]
            input[idx] = (((p shr 16) and 0xFF) - MEAN[0]) / STD[0]
            input[plane + idx] = (((p shr 8) and 0xFF) - MEAN[1]) / STD[1]
            input[2 * plane + idx] = ((p and 0xFF) - MEAN[2]) / STD[2]
        }
    }

    /** Softmax over the logits, then the [k] highest-probability classes. */
    private fun topK(logits: FloatArray, k: Int): List<Prediction> {
        var max = logits[0]
        for (v in logits) {
            if (v > max) {
                max = v
            }
        }
        var sum = 0f
        val probs = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = Math.exp((logits[i] - max).toDouble()).toFloat()
            probs[i] = e
            sum += e
        }
        return (0 until k).map {
            var best = 0
            for (i in probs.indices) {
                if (probs[i] > probs[best]) {
                    best = i
                }
            }
            val p = probs[best] / sum
            probs[best] = -1f
            Prediction(labels.getOrElse(best) { "class $best" }, p)
        }
    }

    override fun close() {
        model.close()
    }
}
