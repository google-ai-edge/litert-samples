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

package com.google.aiedge.examples.zeroshotimageclassification

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.min

/**
 * Runs zero-shot image classification on the LiteRT Compiled Model API with a selectable image
 * tower: Perception Encoder (PE-Core-B16-224, 1024-d) or SigLIP 2 (ViT-B/16, 768-d).
 *
 * The image encoder is converted with litert-torch and runs fully on the GPU delegate (LITERT_CL,
 * no CPU fallback). It maps an image to an L2-normalized embedding. Text embeddings for the
 * candidate labels are pre-computed on the host with the matching text encoder (prompt: "a photo
 * of a {label}") and bundled as a per-model binary asset, so no text model runs on device. The
 * predicted label is the one whose text embedding has the highest cosine similarity to the image
 * embedding.
 */
class ZeroShotImageClassificationHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** Model variant to run. */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations. */
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
        /** Number of top labels to surface. */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Probability below which a label is hidden from the display. */
        var probabilityThreshold: Float = DEFAULT_THRESHOLD,
        /** Kept for API parity with other samples; the Compiled Model API manages threading. */
        var threadCount: Int = DEFAULT_THREAD_COUNT,
    )

    companion object {
        private const val TAG = "ZeroShotImageClassification"

        private const val IMAGE_SIZE = 224
        private const val LABELS_FILE = "labels.txt"

        // PE-Core image normalization: (pixel/255 - 0.5) / 0.5  ->  [-1, 1].
        // Expressed for pixel values in [0, 255]: (pixel - 127.5) / 127.5.
        private val MEAN = floatArrayOf(127.5f, 127.5f, 127.5f)
        private val STD = floatArrayOf(127.5f, 127.5f, 127.5f)

        // PE-Core logit scale (exp of the trained temperature, ~99.9) to sharpen the softmax.
        private const val LOGIT_SCALE = 100f

        val DEFAULT_MODEL = Model.PECoreB16
        val DEFAULT_DELEGATE = AcceleratorEnum.GPU
        const val DEFAULT_RESULT_COUNT = 5
        const val DEFAULT_THRESHOLD = 0.3f
        const val DEFAULT_THREAD_COUNT = 2

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
            }
        }
    }

    val classification: SharedFlow<ClassificationResult>
        get() = _classification
    private val _classification = MutableSharedFlow<ClassificationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var model: CompiledModel? = null
    private var labels: List<String> = emptyList()
    private var textEmbeddings: FloatArray = FloatArray(0)
    private var numLabels: Int = 0
    private var embedDim: Int = 0

    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    // Reusable preprocessing buffers (allocated once).
    private val inputFloats = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
    private val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
    private val squareBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    private val cropMatrix = Matrix()
    private val cropPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Create a [CompiledModel] for the selected [Model] on the selected [AcceleratorEnum]. */
    suspend fun initClassifier() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                loadTextEmbeddings()  // reloads per-model embeddings (models differ in embed space)
                model = CompiledModel.create(
                    context.assets,
                    options.model.fileName,
                    CompiledModel.Options(toAccelerator(options.delegate)),
                    null
                )
                Log.i(TAG, "Created CompiledModel from ${options.model.fileName}")
            }
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Failed to create CompiledModel from ${options.model.fileName}: ${e.message}")
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            model?.close()
            model = null
        }
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    /** Encode [bitmap], score it against every label, and emit the top results. */
    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(singleThreadDispatcher) {
                val currentModel = model ?: return@withContext
                val startTime = SystemClock.uptimeMillis()

                preprocess(bitmap, rotationDegrees)

                val inputBuffers = currentModel.createInputBuffers()
                val outputBuffers = currentModel.createOutputBuffers()
                inputBuffers[0].writeFloat(inputFloats)
                currentModel.run(inputBuffers, outputBuffers)
                val imageEmbedding = outputBuffers[0].readFloat()
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                val categories = scoreLabels(imageEmbedding)
                    .map {
                        if (it.score < options.probabilityThreshold) it.copy(score = 0f) else it
                    }
                    .take(options.resultCount)

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _classification.emit(ClassificationResult(categories, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Zero-shot classification error: ${e.message}")
            _error.emit(e)
        }
    }

    /** Center-crop to square, resize to 224, rotate, then write planar NCHW normalized floats. */
    private fun preprocess(bitmap: Bitmap, rotationDegrees: Int) {
        val side = min(bitmap.width, bitmap.height)
        val left = (bitmap.width - side) / 2f
        val top = (bitmap.height - side) / 2f

        cropMatrix.setRectToRect(
            RectF(left, top, left + side, top + side),
            RectF(0f, 0f, IMAGE_SIZE.toFloat(), IMAGE_SIZE.toFloat()),
            Matrix.ScaleToFit.FILL
        )
        // Apply the camera rotation around the 224x224 center so the model sees an upright image.
        if (rotationDegrees % 360 != 0) {
            cropMatrix.postRotate(
                rotationDegrees.toFloat(),
                IMAGE_SIZE / 2f,
                IMAGE_SIZE / 2f
            )
        }

        val canvas = Canvas(squareBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(bitmap, cropMatrix, cropPaint)
        squareBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            inputFloats[i] = (r - MEAN[0]) / STD[0]
            inputFloats[planeSize + i] = (g - MEAN[1]) / STD[1]
            inputFloats[2 * planeSize + i] = (b - MEAN[2]) / STD[2]
        }
    }

    /** Cosine similarity (dot product of L2-normalized vectors) + temperature-scaled softmax. */
    private fun scoreLabels(imageEmbedding: FloatArray): List<Category> {
        val logits = FloatArray(numLabels)
        for (i in 0 until numLabels) {
            var dot = 0f
            val offset = i * embedDim
            for (j in 0 until embedDim) {
                dot += imageEmbedding[j] * textEmbeddings[offset + j]
            }
            logits[i] = dot * LOGIT_SCALE
        }

        val maxLogit = logits.max()
        var sum = 0f
        val probs = FloatArray(numLabels)
        for (i in 0 until numLabels) {
            val e = exp((logits[i] - maxLogit).toDouble()).toFloat()
            probs[i] = e
            sum += e
        }

        return labels.mapIndexed { i, label -> Category(label, probs[i] / sum) }
            .sortedByDescending { it.score }
    }

    /** Load candidate labels and the selected model's pre-computed text embeddings from assets. */
    private fun loadTextEmbeddings() {
        if (labels.isEmpty()) {
            labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
                .filter { it.isNotBlank() }
        }

        val bytes = context.assets.open(options.model.embeddingsFile).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        numLabels = buffer.int
        embedDim = buffer.int
        textEmbeddings = FloatArray(numLabels * embedDim)
        buffer.asFloatBuffer().get(textEmbeddings)

        require(numLabels == labels.size) {
            "Label/embedding count mismatch: $numLabels embeddings vs ${labels.size} labels"
        }
        Log.i(TAG, "Loaded $numLabels labels x $embedDim-d text embeddings")
    }

    enum class AcceleratorEnum {
        CPU, GPU
    }

    enum class Model(val fileName: String, val embeddingsFile: String) {
        PECoreB16("pe_core_base_224_fp16.tflite", "text_embeddings_pecore.bin"),
        SigLIP2B16("siglip2_base_224_fp16.tflite", "text_embeddings_siglip2.bin")
    }

    data class ClassificationResult(
        val categories: List<Category>, val inferenceTime: Long
    )

    data class Category(val label: String, val score: Float)
}
