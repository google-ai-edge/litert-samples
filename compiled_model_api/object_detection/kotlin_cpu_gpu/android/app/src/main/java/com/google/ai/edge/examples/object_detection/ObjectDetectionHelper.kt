/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.object_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * Runs SSDLite320-MobileNetV3-Large object detection with the LiteRT CompiledModel API.
 *
 * Model: ssdlite_mobilenetv3_320_fp16.tflite (torchvision, BSD-3; converted patch-free via
 * litert-torch).
 *  - input : 1 x 3 x 320 x 320 float32, RGB, NCHW, normalized (px / 127.5 - 1) -> [-1, 1].
 *  - output: 12 raw head tensors, (cls, box) per 6 feature levels (H = W = 20,10,5,3,2,1):
 *              cls[i] = [1, 6*91, H, W]   box[i] = [1, 6*4, H, W]   (6 anchors/loc, 91 classes).
 *
 * Only the backbone + heads run on the GPU; the anchor decode + multiclass NMS are done here in
 * Kotlin, so the graph stays GPU-clean. The model's built-in DefaultBoxGenerator + NMS postprocess
 * would lower to GATHER_ND / TOPK / >4D, which the GPU delegate rejects — tapping the raw 4D head
 * conv outputs avoids that. Decode mirrors torchvision SSD.postprocess_detections + BoxCoder.
 */
class ObjectDetectionHelper(private val context: Context) {

  val detections: SharedFlow<DetectionResult>
    get() = _detections

  private val _detections =
    MutableSharedFlow<DetectionResult>(
      extraBufferCapacity = 64,
      onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private var detector: Detector? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  suspend fun initDetector(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU) {
    cleanup()
    try {
      withContext(singleThreadDispatcher) {
        val model =
          CompiledModel.create(
            context.assets,
            MODEL_FILE,
            CompiledModel.Options(toAccelerator(acceleratorEnum)),
            null,
          )
        detector = Detector(model)
        Log.d(TAG, "Created an object detector ($acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.i(TAG, "Create LiteRT from $MODEL_FILE failed: ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        detector?.cleanup()
        detector = null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun detect(bitmap: Bitmap) {
    try {
      withContext(singleThreadDispatcher) {
        detector?.detect(bitmap)?.let { if (isActive) _detections.emit(it) }
      }
    } catch (e: Exception) {
      Log.i(TAG, "Object detect error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Detector(private val model: CompiledModel) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    // NCHW input, RGB, normalized to [-1, 1].
    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)

    // Default boxes (xyxy, 320-space), order level -> row -> col -> anchor. 3234 anchors.
    private val anchors: FloatArray
    private val levelAnchorOffset = IntArray(LEVELS.size)

    init {
      var total = 0
      for (li in LEVELS.indices) {
        levelAnchorOffset[li] = total
        total += LEVELS[li] * LEVELS[li] * ANCHORS_PER_LOC
      }
      anchors = FloatArray(total * 4)
      buildAnchors()
    }

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun detect(bitmap: Bitmap): DetectionResult {
      val image = bitmap.scale(SIZE, SIZE, true) // bilinear stretch, matches torchvision fixed_size
      val start = SystemClock.uptimeMillis()
      preprocess(image)
      inputBuffers[0].writeFloat(inputFloats)
      model.run(inputBuffers, outputBuffers)
      val dets = decode()
      val inferenceTime = SystemClock.uptimeMillis() - start
      return DetectionResult(dets, bitmap.width, bitmap.height, inferenceTime)
    }

    /** Resize -> NCHW RGB float32, normalized (px / 127.5 - 1) (mean = std = 0.5, NOT ImageNet). */
    private fun preprocess(image: Bitmap) {
      val plane = SIZE * SIZE
      image.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      for (i in 0 until plane) {
        val p = pixels[i]
        inputFloats[i] = Color.red(p) / 127.5f - 1f
        inputFloats[plane + i] = Color.green(p) / 127.5f - 1f
        inputFloats[2 * plane + i] = Color.blue(p) / 127.5f - 1f
      }
    }

    /**
     * Decode the 12 raw head outputs into detections (boxes normalized to [0,1]). Mirrors
     * torchvision SSD.postprocess_detections + BoxCoder(10,10,5,5): per anchor, softmax over 91
     * classes -> best non-background -> score threshold -> decode against the default box -> NMS.
     * Output order is (cls, box) per level; NCHW channel = a*91 + k (cls) / a*4 + j (box).
     */
    private fun decode(): List<Detection> {
      val candidates = ArrayList<Detection>()
      for (li in LEVELS.indices) {
        val h = LEVELS[li]
        val hw = h * h
        val cls = outputBuffers[2 * li].readFloat() // [6*91 * hw]
        val box = outputBuffers[2 * li + 1].readFloat() // [6*4 * hw]
        val offset = levelAnchorOffset[li]

        for (p in 0 until hw) {
          for (a in 0 until ANCHORS_PER_LOC) {
            val clsBase = a * NUM_CLASSES * hw + p
            var maxAll = cls[clsBase]
            var bestLogit = Float.NEGATIVE_INFINITY
            var bestK = 1
            var k = 0
            while (k < NUM_CLASSES) {
              val l = cls[clsBase + k * hw]
              if (l > maxAll) maxAll = l
              if (k >= 1 && l > bestLogit) {
                bestLogit = l
                bestK = k
              }
              k++
            }
            // softmax prob <= exp(bestLogit - maxAll) since sum(exp) >= 1 -> cheap reject.
            val probUpperBound = exp(bestLogit - maxAll)
            if (probUpperBound < SCORE_THRESHOLD) continue
            var sum = 0f
            k = 0
            while (k < NUM_CLASSES) {
              sum += exp(cls[clsBase + k * hw] - maxAll)
              k++
            }
            val score = probUpperBound / sum
            if (score < SCORE_THRESHOLD) continue

            val ai = (offset + p * ANCHORS_PER_LOC + a) * 4
            val ax1 = anchors[ai]
            val ay1 = anchors[ai + 1]
            val aw = anchors[ai + 2] - ax1
            val ah = anchors[ai + 3] - ay1
            val acx = ax1 + 0.5f * aw
            val acy = ay1 + 0.5f * ah
            val boxBase = a * 4 * hw + p
            val dx = box[boxBase] / WX
            val dy = box[boxBase + hw] / WY
            val dw = min(box[boxBase + 2 * hw] / WW, BBOX_CLIP)
            val dh = min(box[boxBase + 3 * hw] / WH, BBOX_CLIP)
            val pcx = dx * aw + acx
            val pcy = dy * ah + acy
            val pw = exp(dw) * aw
            val ph = exp(dh) * ah

            // 320-space xyxy -> clip -> normalize to [0,1].
            val left = (pcx - 0.5f * pw).coerceIn(0f, SIZE.toFloat()) / SIZE
            val top = (pcy - 0.5f * ph).coerceIn(0f, SIZE.toFloat()) / SIZE
            val right = (pcx + 0.5f * pw).coerceIn(0f, SIZE.toFloat()) / SIZE
            val bottom = (pcy + 0.5f * ph).coerceIn(0f, SIZE.toFloat()) / SIZE
            candidates.add(Detection(left, top, right, bottom, bestK, LABELS[bestK], score))
          }
        }
      }
      return nms(candidates.sortedByDescending { it.score })
    }

    /** Per-class greedy NMS (torchvision batched_nms equivalent). */
    private fun nms(sorted: List<Detection>): List<Detection> {
      val result = ArrayList<Detection>()
      val active = BooleanArray(sorted.size) { true }
      for (i in sorted.indices) {
        if (!active[i]) continue
        result.add(sorted[i])
        if (result.size >= MAX_DETECTIONS) break
        for (j in i + 1 until sorted.size) {
          if (active[j] && sorted[j].classId == sorted[i].classId && iou(sorted[i], sorted[j]) > IOU_THRESHOLD) {
            active[j] = false
          }
        }
      }
      return result
    }

    private fun iou(a: Detection, b: Detection): Float {
      val x1 = max(a.left, b.left)
      val y1 = max(a.top, b.top)
      val x2 = min(a.right, b.right)
      val y2 = min(a.bottom, b.bottom)
      val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
      val areaA = (a.right - a.left) * (a.bottom - a.top)
      val areaB = (b.right - b.left) * (b.bottom - b.top)
      return inter / (areaA + areaB - inter + 1e-6f)
    }

    /** torchvision DefaultBoxGenerator for 320x320 (matches the export to ~3e-5). */
    private fun buildAnchors() {
      var idx = 0
      for (li in LEVELS.indices) {
        val f = LEVELS[li]
        val wh = whPairs(li)
        for (i in 0 until f) {
          for (j in 0 until f) {
            val cx = (j + 0.5f) / f
            val cy = (i + 0.5f) / f
            for (a in 0 until ANCHORS_PER_LOC) {
              val w = wh[a * 2]
              val hh = wh[a * 2 + 1]
              anchors[idx++] = (cx - 0.5f * w) * SIZE
              anchors[idx++] = (cy - 0.5f * hh) * SIZE
              anchors[idx++] = (cx + 0.5f * w) * SIZE
              anchors[idx++] = (cy + 0.5f * hh) * SIZE
            }
          }
        }
      }
    }

    private fun whPairs(level: Int): FloatArray {
      val sk = SCALES[level]
      val skp = sqrt(SCALES[level] * SCALES[level + 1])
      val out = ArrayList<Float>(12)
      out.add(sk); out.add(sk)
      out.add(skp); out.add(skp)
      for (ar in ASPECT_RATIOS) {
        val sq = sqrt(ar.toFloat())
        out.add(sk * sq); out.add(sk / sq)
        out.add(sk / sq); out.add(sk * sq)
      }
      return FloatArray(out.size) { out[it].coerceIn(0f, 1f) }
    }
  }

  data class DetectionResult(
    val detections: List<Detection>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceTime: Long,
  )

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  companion object {
    const val TAG = "ObjectDetection"
    const val MODEL_FILE = "ssdlite_mobilenetv3_320_fp16.tflite"
    const val SIZE = 320
    const val NUM_CLASSES = 91 // 90 COCO + background at index 0
    const val ANCHORS_PER_LOC = 6
    val LEVELS = intArrayOf(20, 10, 5, 3, 2, 1)

    // DefaultBoxGenerator(min_ratio=0.2, max_ratio=0.95) -> 7 scales.
    private val SCALES = floatArrayOf(0.2f, 0.35f, 0.5f, 0.65f, 0.8f, 0.95f, 1.0f)
    private val ASPECT_RATIOS = intArrayOf(2, 3)

    // BoxCoder weights (dx, dy, dw, dh) and exp() input clamp.
    private const val WX = 10f
    private const val WY = 10f
    private const val WW = 5f
    private const val WH = 5f
    private val BBOX_CLIP = ln(1000f / 16f)

    private const val SCORE_THRESHOLD = 0.4f
    private const val IOU_THRESHOLD = 0.55f // torchvision SSD nms_thresh
    private const val MAX_DETECTIONS = 100

    fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> Accelerator.CPU
        AcceleratorEnum.GPU -> Accelerator.GPU
      }

    // torchvision COCO label map (91 entries, index 0 = background). "N/A" = unused COCO id gaps.
    private val LABELS =
      arrayOf(
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
        "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
      )
  }
}
