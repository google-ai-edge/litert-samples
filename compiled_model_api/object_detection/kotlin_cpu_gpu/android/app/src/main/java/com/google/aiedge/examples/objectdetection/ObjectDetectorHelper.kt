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

package com.google.aiedge.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
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
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOX object detector on LiteRT CompiledModel (CPU / GPU).
 *
 * The model is the Megvii YOLOX (Apache-2.0) re-authored to a GPU-native TFLite via the official
 * `litert_torch` path: the Focus stem is folded into a single 6x6 stride-2 conv so the graph has
 * zero GATHER_ND / TopK / Cast and no >4D tensors (full LITERT_CL residency on the GPU delegate).
 *
 *   input : images [1, S, S, 3]  NHWC, BGR, 0-255, NO normalization (letterboxed, gray 114 pad)
 *   output: [1, A, 85]           raw heads, anchor-major (A anchors, 85 = 4 box + 1 obj + 80 cls)
 *
 * The graph applies sigmoid to obj/class but does NOT decode boxes, so the grid + stride decode
 * (and per-class NMS) is done here on the host.
 */
class ObjectDetectorHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** The model variant (file name + input size), relative to the assets/ directory. */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running the model (CPU or GPU). */
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
        /** Score (obj * class) above which a detection is kept. */
        var threshold: Float = DEFAULT_THRESHOLD,
    )

    companion object {
        private const val TAG = "ObjectDetection"

        val DEFAULT_MODEL = Model.YoloxS
        val DEFAULT_DELEGATE = AcceleratorEnum.GPU
        const val DEFAULT_THRESHOLD = 0.30f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 50
        private const val PAD_VALUE = 114 // YOLOX letterbox gray
        private val STRIDES = intArrayOf(8, 16, 32)

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
            }
        }
    }

    val detection: SharedFlow<DetectionResult>
        get() = _detection
    private val _detection = MutableSharedFlow<DetectionResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var model: CompiledModel? = null
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    // Per-anchor grid origin + stride, precomputed once per model input size.
    private var gridX = IntArray(0)
    private var gridY = IntArray(0)
    private var gridStride = IntArray(0)
    private var inputSize = 0

    // Reusable preprocessing buffers (re-allocated when the input size changes).
    private var inputFloats = FloatArray(0)
    private var inputPixels = IntArray(0)
    private var letterbox: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Create a CompiledModel for [Options.model] on [Options.delegate] and prepare buffers. */
    suspend fun initDetector() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                inputSize = options.model.inputSize
                buildGrids(inputSize)
                inputFloats = FloatArray(inputSize * inputSize * 3)
                inputPixels = IntArray(inputSize * inputSize)
                letterbox?.recycle()
                letterbox = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

                model = CompiledModel.create(
                    context.assets,
                    options.model.fileName,
                    CompiledModel.Options(toAccelerator(options.delegate)),
                    null,
                )
                Log.i(TAG, "Created CompiledModel ${options.model.fileName} on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Create CompiledModel ${options.model.fileName} failed: ${e.message}")
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

    private fun buildGrids(size: Int) {
        val gx = ArrayList<Int>()
        val gy = ArrayList<Int>()
        val gs = ArrayList<Int>()
        for (stride in STRIDES) {
            val n = size / stride
            for (yy in 0 until n) for (xx in 0 until n) {
                gx.add(xx)
                gy.add(yy)
                gs.add(stride)
            }
        }
        gridX = gx.toIntArray()
        gridY = gy.toIntArray()
        gridStride = gs.toIntArray()
    }

    /**
     * Detect objects in [bitmap]. [rotationDegrees] (from CameraX) is applied first so detections
     * are in upright-image coordinates; the emitted result carries the upright image size so the UI
     * can map boxes onto a fit-center preview.
     */
    suspend fun detect(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(singleThreadDispatcher) {
                val currentModel = model ?: return@withContext
                val startTime = SystemClock.uptimeMillis()

                val upright = rotate(bitmap, rotationDegrees)
                val ratio = min(
                    inputSize.toFloat() / upright.width,
                    inputSize.toFloat() / upright.height,
                )
                writeLetterboxBgr(upright, ratio)

                val inputBuffers = currentModel.createInputBuffers()
                val outputBuffers = currentModel.createOutputBuffers()
                inputBuffers[0].writeFloat(inputFloats)
                currentModel.run(inputBuffers, outputBuffers)
                val raw = outputBuffers[0].readFloat()
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                val detections = postProcess(raw, ratio, upright.width, upright.height)
                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _detection.emit(
                        DetectionResult(detections, inferenceTime, upright.width, upright.height)
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}")
            _error.emit(e)
        }
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return bitmap
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    /** Letterbox (uniform scale, top-left, gray 114 pad) into the NHWC BGR 0-255 input buffer. */
    private fun writeLetterboxBgr(src: Bitmap, ratio: Float) {
        val lb = letterbox ?: return
        val canvas = Canvas(lb)
        canvas.drawColor(Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        val m = Matrix().apply { setScale(ratio, ratio) }
        canvas.drawBitmap(src, m, paint)
        lb.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = (pixel and 0xFF).toFloat()           // B
            inputFloats[idx++] = ((pixel shr 8) and 0xFF).toFloat()   // G
            inputFloats[idx++] = ((pixel shr 16) and 0xFF).toFloat()  // R
        }
    }

    private fun postProcess(raw: FloatArray, ratio: Float, w: Int, h: Int): List<Detection> {
        val candidates = ArrayList<Detection>()
        val threshold = options.threshold
        for (i in gridX.indices) {
            val base = i * 85
            val obj = raw[base + 4]
            if (obj < threshold) continue // class score only lowers it -> early reject

            var bestCls = 0
            var bestScore = 0f
            for (c in 0 until 80) {
                val s = raw[base + 5 + c]
                if (s > bestScore) {
                    bestScore = s
                    bestCls = c
                }
            }
            val score = obj * bestScore
            if (score < threshold) continue

            val stride = gridStride[i]
            val cx = (raw[base] + gridX[i]) * stride
            val cy = (raw[base + 1] + gridY[i]) * stride
            val bw = exp(raw[base + 2]) * stride
            val bh = exp(raw[base + 3]) * stride
            candidates.add(
                Detection(
                    classId = bestCls,
                    label = CocoLabels.name(bestCls),
                    score = score,
                    // un-letterbox (divide by ratio) -> upright-image pixel coords
                    xMin = ((cx - bw / 2f) / ratio).coerceIn(0f, w.toFloat()),
                    yMin = ((cy - bh / 2f) / ratio).coerceIn(0f, h.toFloat()),
                    xMax = ((cx + bw / 2f) / ratio).coerceIn(0f, w.toFloat()),
                    yMax = ((cy + bh / 2f) / ratio).coerceIn(0f, h.toFloat()),
                )
            )
        }
        return nms(candidates.sortedByDescending { it.score }, IOU_THRESHOLD).take(MAX_DETECTIONS)
    }

    /** Per-class non-maximum suppression. */
    private fun nms(sorted: List<Detection>, iouThresh: Float): List<Detection> {
        val result = ArrayList<Detection>()
        val active = BooleanArray(sorted.size) { true }
        for (i in sorted.indices) {
            if (!active[i]) continue
            result.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (active[j] && sorted[j].classId == sorted[i].classId &&
                    iou(sorted[i], sorted[j]) > iouThresh
                ) {
                    active[j] = false
                }
            }
        }
        return result
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = max(a.xMin, b.xMin)
        val y1 = max(a.yMin, b.yMin)
        val x2 = min(a.xMax, b.xMax)
        val y2 = min(a.yMax, b.yMax)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin)
        val areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    enum class AcceleratorEnum { CPU, GPU }

    /** YOLOX variants (Apache-2.0). File names match the litert-community HF downloads. */
    enum class Model(val fileName: String, val inputSize: Int) {
        YoloxNano("yolox_nano.tflite", 416),
        YoloxTiny("yolox_tiny.tflite", 416),
        YoloxS("yolox_s.tflite", 640),
        YoloxM("yolox_m.tflite", 640),
    }

    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val imageWidth: Int,
        val imageHeight: Int,
    )
}
