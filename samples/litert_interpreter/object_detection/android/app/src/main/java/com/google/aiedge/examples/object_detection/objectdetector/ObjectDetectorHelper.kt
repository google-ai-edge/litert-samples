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

package com.google.aiedge.examples.object_detection.objectdetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer

class ObjectDetectorHelper(
    val context: Context,
    var threshold: Float = THRESHOLD_DEFAULT,
    var maxResults: Int = MAX_RESULTS_DEFAULT,
    var delegate: Delegate = Delegate.CPU,
    var model: Model = MODEL_DEFAULT,
) {

    private var interpreter: Interpreter? = null
    private lateinit var labels: List<String>

    private val _detectionResult = MutableSharedFlow<DetectionResult>()
    val detectionResult: SharedFlow<DetectionResult> = _detectionResult

    private val _error = MutableSharedFlow<Throwable>()
    val error: SharedFlow<Throwable> = _error

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    suspend fun setupObjectDetector() {
        try {
            val litertBuffer = FileUtil.loadMappedFile(context, model.fileName)
            labels = getModelMetadata(litertBuffer)
            interpreter = Interpreter(litertBuffer, Interpreter.Options().apply {
                numThreads = 5
                useNNAPI = delegate == Delegate.NNAPI
            })
            Log.i(TAG, "Successfully init Interpreter!")
        } catch (e: Exception) {
            _error.emit(e)
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    // Runs object detection on live streaming cameras frame-by-frame and returns the results
    // asynchronously to the caller.
    suspend fun detect(bitmap: Bitmap, rotationDegrees: Int) {
        if (interpreter == null) return
        withContext(Dispatchers.IO) {
            val startTime = SystemClock.uptimeMillis()
            val (_, h, w, _) = interpreter!!.getInputTensor(0).shape()

            // Preprocess the image and convert it into a TensorImage for classification.
            val tensorImage = createTensorImage(
                bitmap = bitmap, width = w, height = h, rotationDegrees = rotationDegrees
            )

            val output = detectImage(tensorImage)

            val locationOutput = output[0]
            val categoryOutput = output[1]
            val scoreOutput = output[2]
            val detections = getDetections(
                locations = locationOutput,
                categories = categoryOutput,
                scores = scoreOutput,
                width = w,
                scaleRatio = h.toFloat() / tensorImage.height
            )
            val inferenceTime = SystemClock.uptimeMillis() - startTime

            val detectionResult = DetectionResult(
                detections = detections,
                inferenceTime = inferenceTime,
                inputImageWidth = w,
                inputImageHeight = h,
            )
            _detectionResult.emit(detectionResult)
        }
    }

    suspend fun detect(imageProxy: ImageProxy) {
        detect(
            bitmap = imageProxy.toBitmap(), rotationDegrees = imageProxy.imageInfo.rotationDegrees
        )
    }

    private fun detectImage(tensorImage: TensorImage): List<FloatArray> {
        val locationOutputShape = interpreter!!.getOutputTensor(0).shape()
        val categoryOutputShape = interpreter!!.getOutputTensor(1).shape()
        val scoreOutputShape = interpreter!!.getOutputTensor(2).shape()

        val locationOutputBuffer =
            FloatBuffer.allocate(locationOutputShape[1] * locationOutputShape[2])
        val scoreOutputBuffer = FloatBuffer.allocate(scoreOutputShape[1])
        val categoryOutputBuffer = FloatBuffer.allocate(categoryOutputShape[1])

        locationOutputBuffer.rewind()
        scoreOutputBuffer.rewind()
        categoryOutputBuffer.rewind()
        interpreter?.runForMultipleInputsOutputs(
            arrayOf(tensorImage.tensorBuffer.buffer), mapOf(
                Pair(0, locationOutputBuffer),
                Pair(1, categoryOutputBuffer),
                Pair(2, scoreOutputBuffer),
            )
        )

        locationOutputBuffer.rewind()
        scoreOutputBuffer.rewind()
        categoryOutputBuffer.rewind()
        val locationOutput = FloatArray(locationOutputBuffer.capacity())
        val scoreOutput = FloatArray(scoreOutputBuffer.capacity())
        val categoryOutput = FloatArray(categoryOutputBuffer.capacity())

        locationOutputBuffer.get(locationOutput)
        scoreOutputBuffer.get(scoreOutput)
        categoryOutputBuffer.get(categoryOutput)

        return listOf(locationOutput, categoryOutput, scoreOutput)
    }

    private fun createTensorImage(
        bitmap: Bitmap, width: Int, height: Int, rotationDegrees: Int
    ): TensorImage {
        val rotation = -rotationDegrees / 90
        val scaledBitmap = fitCenterBitmap(bitmap, width, height)

        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(rotation)).build()

        // Preprocess the image and convert it into a TensorImage for classification.
        return imageProcessor.process(TensorImage.fromBitmap(scaledBitmap))
    }

    private fun fitCenterBitmap(
        originalBitmap: Bitmap,
        width: Int,
        height: Int
    ): Bitmap {
        val bitmapWithBackground = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmapWithBackground)
        canvas.drawColor(Color.TRANSPARENT)

        val scale: Float = height.toFloat() / originalBitmap.height
        val dstWidth = width * scale
        val processBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val scaledBitmap = Bitmap.createScaledBitmap(
            processBitmap, dstWidth.toInt(), height, true
        )

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        val left = (width - dstWidth) / 2
        canvas.drawBitmap(scaledBitmap, left, 0f, paint)
        return bitmapWithBackground
    }

    /** Load metadata from model*/
    private fun getModelMetadata(litertBuffer: ByteBuffer): List<String> {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        val labels = mutableListOf<String>()
        if (metadataExtractor.hasMetadata()) {
            val inputStream = metadataExtractor.getAssociatedFile("labelmap.txt")
            labels.addAll(readFileInputStream(inputStream))
            Log.i(
                TAG, "Successfully loaded model metadata ${metadataExtractor.associatedFileNames}"
            )
        }
        return labels
    }

    /** Retrieve Map<String, Int> from metadata file */
    private fun readFileInputStream(inputStream: InputStream): List<String> {
        val reader = BufferedReader(InputStreamReader(inputStream))

        val list = mutableListOf<String>()
        var index = 0
        var line = ""
        while (reader.readLine().also { if (it != null) line = it } != null) {
            list.add(line)
            index++
        }

        reader.close()
        return list
    }

    private fun getDetections(
        locations: FloatArray,
        categories: FloatArray,
        scores: FloatArray,
        width: Int,
        scaleRatio: Float
    ): List<Detection> {
        val boundingBoxList = getBoundingBoxList(locations, width, scaleRatio)

        val detections = mutableListOf<Detection>()
        for (i in 0..<maxResults) {
            val categoryIndex = categories[i].toInt()
            detections.add(
                Detection(
                    label = labels[categoryIndex],
                    boundingBox = boundingBoxList[i],
                    score = scores[i]
                )
            )
        }

        return detections
            .filter { !it.boundingBox.isEmpty && it.score >= THRESHOLD_DEFAULT }
            .sortedByDescending { it.score }
    }

    /**
     * A tf.float32 tensor of shape [N, 4] containing bounding box coordinates
     * in the following order: [ymin, xmin, ymax, xmax]
     */
    private fun getBoundingBoxList(
        locations: FloatArray, width: Int, scaleRatio: Float
    ): Array<RectF> {
        val boundingBoxList = Array(locations.size / 4) { RectF() }
        val actualWidth = width * scaleRatio
        val padding = (width - width * scaleRatio) / 2

        for (i in boundingBoxList.indices) {
            val topRatio = locations[i * 4]
            val leftRatio = locations[i * 4 + 1]
            val bottomRatio = locations[i * 4 + 2]
            val rightRatio = locations[i * 4 + 3]

            val top = topRatio.coerceAtLeast(0f).coerceAtMost(1f)
            val left =
                ((leftRatio * width - padding) / actualWidth).coerceAtLeast(0f).coerceAtMost(1f)
            val bottom = bottomRatio.coerceAtLeast(top).coerceAtMost(1f)
            val right =
                ((rightRatio * width - padding) / actualWidth).coerceAtLeast(left).coerceAtMost(1f)

            val rectF = RectF(left, top, right, bottom)
            boundingBoxList[i] = rectF
        }

        return boundingBoxList;
    }


    companion object {
        val MODEL_DEFAULT = Model.EfficientDetLite2
        const val MAX_RESULTS_DEFAULT = 5
        const val THRESHOLD_DEFAULT = 0.5F

        const val TAG = "ObjectDetectorHelper"
    }

    enum class Model(val fileName: String) {
        EfficientDetLite2("efficientdet-lite2.tflite"),
    }

    enum class Delegate(val value: Int) {
        CPU(0), NNAPI(1)
    }


    // Wraps results from inference, the time it takes for inference to be performed, and
    // the input image and height for properly scaling UI to return back to callers
    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    data class Detection(
        val label: String, val boundingBox: RectF, val score: Float
    )
}
