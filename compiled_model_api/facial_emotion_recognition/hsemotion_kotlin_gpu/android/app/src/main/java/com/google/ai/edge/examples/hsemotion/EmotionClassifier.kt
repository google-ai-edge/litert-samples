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

package com.google.ai.edge.examples.hsemotion

import android.content.Context
import android.graphics.Bitmap
import android.media.FaceDetector
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * Facial emotion recognition with HSEmotion (EfficientNet-B0, AffectNet) on the
 * LiteRT CompiledModel GPU. The whole network runs on the GPU delegate. The only
 * host work is a face crop and ImageNet normalization.
 *
 * The model expects a tightly cropped face, so [classify] first locates a face
 * with the built-in [FaceDetector] and crops to it (falling back to the whole
 * image when no face is found), then resizes to 224x224 and normalizes.
 *
 * Input : [1, 3, 224, 224] NCHW, RGB, ImageNet-normalized.
 * Output: [1, 8] emotion logits (index order in [EMOTIONS]).
 */
class EmotionClassifier(context: Context) : AutoCloseable {

    companion object {
        const val SIZE = 224
        val EMOTIONS = arrayOf(
            "Anger", "Contempt", "Disgust", "Fear",
            "Happiness", "Neutral", "Sadness", "Surprise")
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val model = CompiledModel.create(
        context.assets, "hsemotion_b0_fp16.tflite",
        CompiledModel.Options(Accelerator.GPU), null)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()
    private val input = FloatArray(3 * SIZE * SIZE)

    /** One prediction: emotion label plus its softmax probability. */
    data class Prediction(val label: String, val probability: Float)

    /** Detects the face in [bitmap], classifies it, returns emotions sorted. */
    fun classify(bitmap: Bitmap): List<Prediction> {
        val face = cropFace(bitmap)
        preprocess(face)
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return softmaxSorted(outputs[0].readFloat())
    }

    /**
     * Returns the face region of [bitmap] using the platform face detector, or
     * the whole bitmap when no face is found. The crop is a square centered on
     * the face, sized from the detected eye distance.
     */
    private fun cropFace(bitmap: Bitmap): Bitmap {
        val width = bitmap.width and 1.inv()   // FaceDetector needs an even width
        val rgb565 = Bitmap.createBitmap(width, bitmap.height, Bitmap.Config.RGB_565)
        android.graphics.Canvas(rgb565).drawBitmap(bitmap, 0f, 0f, null)
        val faces = arrayOfNulls<FaceDetector.Face>(1)
        val found = FaceDetector(width, bitmap.height, 1).findFaces(rgb565, faces)
        if (found == 0) {
            return bitmap
        }
        val face = faces[0] ?: return bitmap
        val mid = android.graphics.PointF()
        face.getMidPoint(mid)
        val half = face.eyesDistance() * 1.6f
        val left = (mid.x - half).toInt().coerceIn(0, bitmap.width - 1)
        val top = (mid.y - half).toInt().coerceIn(0, bitmap.height - 1)
        val right = (mid.x + half).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (mid.y + half * 1.2f).toInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    /** Resize to 224 -> ImageNet-normalized NCHW into [input]. */
    private fun preprocess(face: Bitmap) {
        val resized = Bitmap.createScaledBitmap(face, SIZE, SIZE, true)
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

    /** Softmax over the 8 logits, returned highest-probability first. */
    private fun softmaxSorted(logits: FloatArray): List<Prediction> {
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
        return EMOTIONS.indices
            .map { Prediction(EMOTIONS[it], probs[it] / sum) }
            .sortedByDescending { it.probability }
    }

    override fun close() {
        model.close()
    }
}
