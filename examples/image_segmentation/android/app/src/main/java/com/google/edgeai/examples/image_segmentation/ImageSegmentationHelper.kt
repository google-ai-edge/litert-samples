/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.edgeai.examples.image_segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ImageProperties
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Random

class ImageSegmentationHelper(private val context: Context) {
    companion object {
        private const val TAG = "ImageSegmentation"
    }

    /** As the result of sound classification, this value emits map of probabilities */
    val segmentation: SharedFlow<SegmentationResult>
        get() = _segmentation
    private val _segmentation = MutableSharedFlow<SegmentationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var interpreter: Interpreter? = null
    private val coloredLabels: List<ColoredLabel> = coloredLabels()

    /** Init a Interpreter from deeplap_v3.*/
    suspend fun initClassifier(delegate: Delegate = Delegate.CPU) {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, "deeplab_v3.tflite")
            Log.i(TAG, "Done creating LiteRT buffer from deeplab_v3")
            val options = Interpreter.Options().apply {
                numThreads = 4
                useNNAPI = delegate == Delegate.NNAPI
            }
            Interpreter(litertBuffer, options)
        } catch (e: Exception) {
            Log.i(TAG, "Create LiteRT from deeplab_v3 is failed ${e.message}")
            _error.emit(e)
            null
        }
    }

    suspend fun segment(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(Dispatchers.IO) {
                if (interpreter == null) return@withContext
                val startTime = SystemClock.uptimeMillis()

                val rotation = -rotationDegrees / 90
                val (_, h, w, _) = interpreter?.getOutputTensor(0)?.shape() ?: return@withContext
                val imageProcessor =
                    ImageProcessor
                        .Builder()
                        .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
                        .add(Rot90Op(rotation))
                        .add(NormalizeOp(127.5f, 127.5f))
                        .build()

                // Preprocess the image and convert it into a TensorImage for segmentation.
                val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
                val segmentResult = segment(tensorImage)
                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _segmentation.emit(SegmentationResult(segmentResult, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image segment error occurred: ${e.message}")
            _error.emit(e)
        }
    }

    private fun segment(tensorImage: TensorImage): Segmentation {
        val (_, h, w, c) = interpreter!!.getOutputTensor(0).shape()
        val outputBuffer = FloatBuffer.allocate(h * w * c)

        outputBuffer.rewind()
        interpreter?.run(tensorImage.tensorBuffer.buffer, outputBuffer)

        outputBuffer.rewind()
        val inferenceData =
            InferenceData(width = w, height = h, channels = c, buffer = outputBuffer)
        val mask = processImage(inferenceData)

        val imageProperties =
            ImageProperties
                .builder()
                .setWidth(inferenceData.width)
                .setHeight(inferenceData.height)
                .setColorSpaceType(ColorSpaceType.GRAYSCALE)
                .build()
        val maskImage = TensorImage()
        maskImage.load(mask, imageProperties)
        return Segmentation(
            listOf(maskImage), coloredLabels
        )
    }

    private fun processImage(inferenceData: InferenceData): ByteBuffer {
        val mask = ByteBuffer.allocateDirect(inferenceData.width * inferenceData.height)
        for (i in 0 until inferenceData.height) {
            for (j in 0 until inferenceData.width) {
                val offset = inferenceData.channels * (i * inferenceData.width + j)

                var maxIndex = 0
                var maxValue = inferenceData.buffer.get(offset)

                for (index in 1 until inferenceData.channels) {
                    if (inferenceData.buffer.get(offset + index) > maxValue) {
                        maxValue = inferenceData.buffer.get(offset + index)
                        maxIndex = index
                    }
                }

                mask.put(i * inferenceData.width + j, maxIndex.toByte())
            }
        }

        return mask
    }

    private fun coloredLabels(): List<ColoredLabel> {
        val labels = listOf(
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv",
            "------"
        )
        val colors = MutableList(labels.size) {
            ColoredLabel(
                labels[0], "", Color.BLACK
            )
        }

        val random = Random()
        val goldenRatioConjugate = 0.618033988749895
        var hue = random.nextDouble()

        // Skip the first label as it's already assigned black
        for (idx in 1 until labels.size) {
            hue += goldenRatioConjugate
            hue %= 1.0
            // Adjust saturation & lightness as needed
            val color = Color.HSVToColor(floatArrayOf(hue.toFloat() * 360, 0.7f, 0.8f))
            colors[idx] = ColoredLabel(labels[idx], "", color)
        }

        return colors
    }

    data class Segmentation(
        val masks: List<TensorImage>,
        val coloredLabels: List<ColoredLabel>,
    )

    data class ColoredLabel(val label: String, val displayName: String, val argb: Int)

    enum class Delegate {
        CPU, NNAPI
    }

    data class SegmentationResult(
        val segmentation: Segmentation,
        val inferenceTime: Long
    )

    data class InferenceData(
        val width: Int,
        val height: Int,
        val channels: Int,
        val buffer: FloatBuffer,
    )
}
