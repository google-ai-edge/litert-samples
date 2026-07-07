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
import com.google.ai.edge.litert.TensorBuffer
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
 * Object detector on LiteRT CompiledModel (CPU / GPU) with two selectable detector families:
 *
 * YOLOX (Megvii, Apache-2.0) — a single-graph pure CNN, re-authored to a GPU-native TFLite via the
 * official `litert_torch` path: the Focus stem is folded into a single 6x6 stride-2 conv so the
 * graph has zero GATHER_ND / TopK / Cast and no >4D tensors (full LITERT_CL residency).
 *
 *   input : images [1, S, S, 3]  NHWC, BGR, 0-255, NO normalization (letterboxed, gray 114 pad)
 *   output: [1, A, 85]           raw heads, anchor-major (A anchors, 85 = 4 box + 1 obj + 80 cls)
 *
 * The graph applies sigmoid to obj/class but does NOT decode boxes, so the grid + stride decode
 * (and per-class NMS) is done here on the host.
 *
 * RF-DETR Nano (Roboflow, Apache-2.0) — a two-stage transformer DETR whose query selection
 * (topk + gather) has no GPU-compatible op, so it ships as TWO GPU graphs with a tiny host step
 * between them (the standard two-stage-DETR edge split):
 *
 *   Graph A (GPU)  image[1,3,384,384] -> enc_class[1,576,91], enc_coord[1,576,4], memory[1,576,256]
 *   host (here)    topk-300 by max class score -> gather enc_coord -> refpoint_ts[1,300,4]
 *   Graph B (GPU)  (memory, refpoint_ts) -> boxes[1,300,4] (cxcywh, [0,1]), logits[1,300,91]
 *   host (here)    sigmoid + score threshold + cxcywh->xyxy + per-class NMS
 *
 * RF-DETR preprocessing is a plain square resize (no letterbox) to 384x384, RGB, ImageNet
 * mean/std, NCHW. Its logits use the 91-way COCO category-id space (mapped back to the contiguous
 * 80-class space via [CocoLabels.index80]).
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

        // RF-DETR two-graph split (see the class doc).
        private const val RFDETR_NPROP = 576 // 24x24 proposal grid
        private const val RFDETR_NQ = 300 // decoder queries
        private const val RFDETR_NCLS = 91 // COCO category-id space (index == COCO category id)
        private const val RFDETR_HID = 256
        private const val RFDETR_IOU_THRESHOLD = 0.6f // light NMS — cleans fp16 near-duplicate queries
        private val RFDETR_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val RFDETR_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

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

    private var model: CompiledModel? = null // YOLOX graph, or RF-DETR graph A
    private var modelB: CompiledModel? = null // RF-DETR graph B (decoder); null for single-graph models
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    // Per-anchor grid origin + stride, precomputed once per model input size (YOLOX only).
    private var gridX = IntArray(0)
    private var gridY = IntArray(0)
    private var gridStride = IntArray(0)
    private var inputSize = 0

    // RF-DETR: persistent IO buffers (created once per init) with slots resolved by float capacity,
    // robust to converter input/output ordering.
    private var rfInA: List<TensorBuffer> = emptyList()
    private var rfOutA: List<TensorBuffer> = emptyList()
    private var rfInB: List<TensorBuffer> = emptyList()
    private var rfOutB: List<TensorBuffer> = emptyList()
    private var slotAEncClass = 0
    private var slotAEncCoord = 0
    private var slotAMemory = 0
    private var slotBMemory = 0
    private var slotBRefpoints = 0
    private var slotBBoxes = 0
    private var slotBLogits = 0

    // Reusable preprocessing buffers (re-allocated when the input size changes).
    private var inputFloats = FloatArray(0)
    private var inputPixels = IntArray(0)
    private var letterbox: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Create CompiledModel(s) for [Options.model] on [Options.delegate] and prepare buffers. */
    suspend fun initDetector() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                inputSize = options.model.inputSize
                inputFloats = FloatArray(inputSize * inputSize * 3)
                inputPixels = IntArray(inputSize * inputSize)
                letterbox?.recycle()
                letterbox = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

                val compiledOptions = CompiledModel.Options(toAccelerator(options.delegate))
                model = CompiledModel.create(
                    context.assets,
                    options.model.fileName,
                    compiledOptions,
                    null,
                )

                val graphBFile = options.model.graphBFileName
                if (graphBFile == null) {
                    buildGrids(inputSize)
                } else {
                    val graphB = CompiledModel.create(context.assets, graphBFile, compiledOptions, null)
                    modelB = graphB
                    rfInA = model!!.createInputBuffers()
                    rfOutA = model!!.createOutputBuffers()
                    rfInB = graphB.createInputBuffers()
                    rfOutB = graphB.createOutputBuffers()
                    slotAEncClass =
                        rfOutA.indexOfFirst { it.readFloat().size == RFDETR_NPROP * RFDETR_NCLS }
                    slotAEncCoord = rfOutA.indexOfFirst { it.readFloat().size == RFDETR_NPROP * 4 }
                    slotAMemory =
                        rfOutA.indexOfFirst { it.readFloat().size == RFDETR_NPROP * RFDETR_HID }
                    slotBMemory =
                        rfInB.indexOfFirst { it.readFloat().size == RFDETR_NPROP * RFDETR_HID }
                    slotBRefpoints = rfInB.indexOfFirst { it.readFloat().size == RFDETR_NQ * 4 }
                    slotBBoxes = rfOutB.indexOfFirst { it.readFloat().size == RFDETR_NQ * 4 }
                    slotBLogits =
                        rfOutB.indexOfFirst { it.readFloat().size == RFDETR_NQ * RFDETR_NCLS }
                }
                Log.i(TAG, "Created CompiledModel ${options.model.fileName} on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Create CompiledModel ${options.model.fileName} failed: ${e.message}")
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            (rfInA + rfOutA + rfInB + rfOutB).forEach { it.close() }
            rfInA = emptyList()
            rfOutA = emptyList()
            rfInB = emptyList()
            rfOutB = emptyList()
            modelB?.close()
            modelB = null
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
                val detections = if (options.model.graphBFileName != null) {
                    val decoder = modelB ?: return@withContext
                    runRfDetr(currentModel, decoder, upright)
                } else {
                    runYolox(currentModel, upright)
                }
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

    /** YOLOX: letterbox -> single graph -> grid/stride decode + per-class NMS. */
    private fun runYolox(currentModel: CompiledModel, upright: Bitmap): List<Detection> {
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

        return postProcess(raw, ratio, upright.width, upright.height)
    }

    /** RF-DETR: square resize -> graph A -> host topk/gather -> graph B -> decode + NMS. */
    private fun runRfDetr(
        graphA: CompiledModel,
        graphB: CompiledModel,
        upright: Bitmap,
    ): List<Detection> {
        writeSquareRgbChw(upright)

        // Graph A: backbone + encoder + proposal heads (GPU).
        rfInA[0].writeFloat(inputFloats)
        graphA.run(rfInA, rfOutA)
        val encClass = rfOutA[slotAEncClass].readFloat() // [576*91]
        val encCoord = rfOutA[slotAEncCoord].readFloat() // [576*4]
        val memory = rfOutA[slotAMemory].readFloat() // [576*256]

        // Host: top-300 proposals by max class logit, gather their coords (descending = torch.topk).
        val maxScore = FloatArray(RFDETR_NPROP)
        for (p in 0 until RFDETR_NPROP) {
            var m = -Float.MAX_VALUE
            val base = p * RFDETR_NCLS
            for (c in 0 until RFDETR_NCLS) {
                val v = encClass[base + c]
                if (v > m) m = v
            }
            maxScore[p] = m
        }
        val order = (0 until RFDETR_NPROP).sortedByDescending { maxScore[it] }
        val refpoints = FloatArray(RFDETR_NQ * 4)
        for (i in 0 until RFDETR_NQ) {
            val src = order[i] * 4
            refpoints[i * 4] = encCoord[src]
            refpoints[i * 4 + 1] = encCoord[src + 1]
            refpoints[i * 4 + 2] = encCoord[src + 2]
            refpoints[i * 4 + 3] = encCoord[src + 3]
        }

        // Graph B: two-stage combine + decoder + heads (GPU).
        rfInB[slotBMemory].writeFloat(memory)
        rfInB[slotBRefpoints].writeFloat(refpoints)
        graphB.run(rfInB, rfOutB)
        val boxes = rfOutB[slotBBoxes].readFloat() // [300*4] cxcywh in [0,1]
        val logits = rfOutB[slotBLogits].readFloat() // [300*91]

        // Host: sigmoid + threshold + cxcywh -> xyxy in upright-image pixels + per-class NMS.
        val w = upright.width.toFloat()
        val h = upright.height.toFloat()
        val candidates = ArrayList<Detection>()
        for (q in 0 until RFDETR_NQ) {
            var best = -Float.MAX_VALUE
            var bestId = -1
            val base = q * RFDETR_NCLS
            for (c in 0 until RFDETR_NCLS) {
                val v = logits[base + c]
                if (v > best) {
                    best = v
                    bestId = c
                }
            }
            val score = 1f / (1f + exp(-best))
            if (score < options.threshold) continue
            val classId = CocoLabels.index80(bestId)
            if (classId < 0) continue // background (0) / unused COCO category ids

            // Boxes are normalized in the square-resized space, which maps directly onto the image.
            val cx = boxes[q * 4] * w
            val cy = boxes[q * 4 + 1] * h
            val bw = boxes[q * 4 + 2] * w
            val bh = boxes[q * 4 + 3] * h
            candidates.add(
                Detection(
                    classId = classId,
                    label = CocoLabels.name(classId),
                    score = score,
                    xMin = (cx - bw / 2f).coerceIn(0f, w),
                    yMin = (cy - bh / 2f).coerceIn(0f, h),
                    xMax = (cx + bw / 2f).coerceIn(0f, w),
                    yMax = (cy + bh / 2f).coerceIn(0f, h),
                )
            )
        }
        return nms(candidates.sortedByDescending { it.score }, RFDETR_IOU_THRESHOLD)
            .take(MAX_DETECTIONS)
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

    /** Plain square resize (no letterbox) into the NCHW RGB ImageNet-normalized input buffer. */
    private fun writeSquareRgbChw(src: Bitmap) {
        val lb = letterbox ?: return
        val canvas = Canvas(lb)
        val m = Matrix().apply {
            setScale(inputSize.toFloat() / src.width, inputSize.toFloat() / src.height)
        }
        canvas.drawBitmap(src, m, paint)
        lb.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        val hw = inputSize * inputSize
        for (i in 0 until hw) {
            val pixel = inputPixels[i]
            inputFloats[i] =
                (((pixel shr 16) and 0xFF) / 255f - RFDETR_MEAN[0]) / RFDETR_STD[0] // R
            inputFloats[hw + i] =
                (((pixel shr 8) and 0xFF) / 255f - RFDETR_MEAN[1]) / RFDETR_STD[1] // G
            inputFloats[2 * hw + i] =
                ((pixel and 0xFF) / 255f - RFDETR_MEAN[2]) / RFDETR_STD[2] // B
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

    /**
     * Selectable detectors. File names match the litert-community HF downloads (see
     * `download_model.gradle`). YOLOX variants are single-graph; RF-DETR Nano is the two-graph
     * DETR split (graph A + graph B).
     */
    enum class Model(
        val fileName: String,
        val inputSize: Int,
        val graphBFileName: String? = null,
    ) {
        YoloxNano("yolox_nano.tflite", 416),
        YoloxTiny("yolox_tiny.tflite", 416),
        YoloxS("yolox_s.tflite", 640),
        YoloxM("yolox_m.tflite", 640),
        RfDetrNano("rfdetr_graphA_fp16.tflite", 384, "rfdetr_graphB_fp16.tflite"),
    }

    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val imageWidth: Int,
        val imageHeight: Int,
    )
}
