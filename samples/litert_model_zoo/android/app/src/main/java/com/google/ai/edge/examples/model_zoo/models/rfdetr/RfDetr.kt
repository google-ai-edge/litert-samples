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

package com.google.ai.edge.examples.model_zoo.models.rfdetr

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

/**
 * RF-DETR Nano object detection, fully on the LiteRT CompiledModel GPU (ML Drift / LITERT_CL).
 *
 * RF-DETR is a two-stage DETR whose query selection (topk + gather) has no GPU-compatible op, so
 * the model is split into two GPU graphs with a tiny host step between them — the standard
 * two-stage-DETR edge split:
 *
 * Graph A (GPU) image[1,3,384,384] -> enc_class[1,576,91], enc_coord[1,576,4], memory[1,576,256]
 * host (here) topk-300 by max class score -> gather enc_coord -> refpoint_ts[1,300,4] Graph B (GPU)
 * (memory, refpoint_ts) -> boxes[1,300,4] (cxcywh), logits[1,300,91] host (here) sigmoid + score
 * threshold + cxcywh->xyxy + per-class NMS -> detections
 *
 * Both graphs run 100% on the GPU (1381/1381 and 404/404 nodes LITERT_CL on a Pixel 8a, ~27 ms
 * total). The DINOv2 backbone is windowed (only 3 global-attention layers), and the projector +
 * decoder LayerNorms use a down-scaled fp16-safe reduction, so the whole thing survives the Mali
 * fp16 path.
 */
class RfDetr(private val modelDir: File, private val accelerator: Accelerator = Accelerator.GPU) :
  Closeable {

  private val resources = mutableListOf<AutoCloseable>()

  private fun <T : AutoCloseable> owned(create: () -> T): T =
    try {
      create().also { resources.add(it) }
    } catch (failure: Throwable) {
      close()
      throw failure
    }

  private fun buffers(create: () -> List<TensorBuffer>): List<TensorBuffer> =
    try {
      create().also { resources.addAll(it) }
    } catch (failure: Throwable) {
      close()
      throw failure
    }

  private fun slot(buffers: List<TensorBuffer>, floatCount: Int): Int =
    try {
      buffers
        .indexOfFirst { it.readFloat().size == floatCount }
        .also { check(it >= 0) { "RF-DETR tensor slot with $floatCount floats was not found" } }
    } catch (failure: Throwable) {
      close()
      throw failure
    }

  companion object {
    const val SIZE = 384
    const val MODEL_A = "rfdetr_graphA_fp16.tflite"
    const val MODEL_B = "rfdetr_graphB_fp16.tflite"
    const val NPROP = 576 // 24x24 proposal grid
    const val NQ = 300 // decoder queries
    const val NCLS = 91 // COCO id space (index == COCO category id)
    const val HID = 256
    // ImageNet normalization (RF-DETR preprocessing)
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    const val SCORE_THRESH = 0.45f
    const val IOU_THRESH = 0.6f // light NMS — cleans fp16 near-duplicate queries

    /** The input normalization loop, exposed for JVM fixtures. */
    fun normalizeRgb(rgb: FloatArray): FloatArray {
      val chw = FloatArray(3 * SIZE * SIZE)
      val hw = SIZE * SIZE
      for (i in 0 until hw) {
        chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
        chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
        chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
      }
      return chw
    }

    /** The top-300 proposal selection. */
    fun selectReferencePoints(encClass: FloatArray, encCoord: FloatArray): FloatArray {
      val maxScore = FloatArray(NPROP)
      for (p in 0 until NPROP) {
        var m = -Float.MAX_VALUE
        val base = p * NCLS
        for (c in 0 until NCLS) {
          val v = encClass[base + c]
          if (v > m) m = v
        }
        maxScore[p] = m
      }
      val order = (0 until NPROP).sortedByDescending { maxScore[it] }
      val ts = FloatArray(NQ * 4)
      for (i in 0 until NQ) {
        val src = order[i] * 4
        ts[i * 4] = encCoord[src]
        ts[i * 4 + 1] = encCoord[src + 1]
        ts[i * 4 + 2] = encCoord[src + 2]
        ts[i * 4 + 3] = encCoord[src + 3]
      }
      return ts
    }

    /** Box decode and class-aware suppression. */
    fun decode(boxes: FloatArray, logits: FloatArray): List<Detection> {
      val dets = ArrayList<Detection>()
      for (q in 0 until NQ) {
        var best = -Float.MAX_VALUE
        var bestC = -1
        val base = q * NCLS
        for (c in 0 until NCLS) {
          val v = logits[base + c]
          if (v > best) {
            best = v
            bestC = c
          }
        }
        val score = 1f / (1f + Math.exp(-best.toDouble()).toFloat()) // sigmoid
        if (score < SCORE_THRESH || bestC <= 0) continue // index 0 is unused (background)
        dets.add(
          Detection(
            bestC,
            score,
            boxes[q * 4],
            boxes[q * 4 + 1],
            boxes[q * 4 + 2],
            boxes[q * 4 + 3],
          )
        )
      }
      return nms(dets)
    }

    private fun nms(dets: List<Detection>): List<Detection> {
      val out = ArrayList<Detection>()
      for (cls in dets.map { it.cls }.toSet()) {
        val group = dets.filter { it.cls == cls }.sortedByDescending { it.score }
        val taken = BooleanArray(group.size)
        for (i in group.indices) {
          if (taken[i]) continue
          out.add(group[i])
          for (j in i + 1 until group.size) if (!taken[j] && iou(group[i], group[j]) > IOU_THRESH)
            taken[j] = true
        }
      }
      return out.sortedByDescending { it.score }
    }

    private fun iou(a: Detection, b: Detection): Float {
      val ax0 = a.cx - a.w / 2
      val ay0 = a.cy - a.h / 2
      val ax1 = a.cx + a.w / 2
      val ay1 = a.cy + a.h / 2
      val bx0 = b.cx - b.w / 2
      val by0 = b.cy - b.h / 2
      val bx1 = b.cx + b.w / 2
      val by1 = b.cy + b.h / 2
      val ix0 = maxOf(ax0, bx0)
      val iy0 = maxOf(ay0, by0)
      val ix1 = minOf(ax1, bx1)
      val iy1 = minOf(ay1, by1)
      val iw = maxOf(0f, ix1 - ix0)
      val ih = maxOf(0f, iy1 - iy0)
      val inter = iw * ih
      val ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
      return if (ua > 0f) inter / ua else 0f
    }
  }

  /** Box coords are normalized [0,1] in letterboxed-SIZE space. */
  data class Detection(
    val cls: Int,
    val score: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
  )

  private fun load(name: String): CompiledModel {
    val f = File(modelDir, name)
    check(f.exists()) { "Model not found: $name. Download this task from Models first." }
    return CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
  }

  private val ga = owned { load(MODEL_A) }
  private val gb = owned { load(MODEL_B) }
  private val aIn = buffers { ga.createInputBuffers() }
  private val aOut = buffers { ga.createOutputBuffers() }
  private val bIn = buffers { gb.createInputBuffers() }
  private val bOut = buffers { gb.createOutputBuffers() }

  // Resolve output/input buffer slots by float capacity (robust to converter ordering).
  private val aEncClass = slot(aOut, NPROP * NCLS)
  private val aEncCoord = slot(aOut, NPROP * 4)
  private val aMemory = slot(aOut, NPROP * HID)
  private val bMemSlot = slot(bIn, NPROP * HID)
  private val bTsSlot = slot(bIn, NQ * 4)
  private val bBoxes = slot(bOut, NQ * 4)
  private val bLogits = slot(bOut, NQ * NCLS)

  /**
   * rgb: SIZE*SIZE*3 row-major [0,255]. Returns detections (boxes normalized in [0,1] SIZE space).
   */
  fun detect(rgb: FloatArray): List<Detection> {
    // ---- Graph A: backbone + encoder + proposal heads (GPU) ----
    val chw = normalizeRgb(rgb)
    aIn[0].writeFloat(chw)
    ga.run(aIn, aOut)
    val encClass = aOut[aEncClass].readFloat() // [576*91]
    val encCoord = aOut[aEncCoord].readFloat() // [576*4]
    val memory = aOut[aMemory].readFloat() // [576*256]

    // ---- host: topk-300 by max class score, gather coords (descending order, matches torch.topk)
    // ----
    val ts = selectReferencePoints(encClass, encCoord)

    // ---- Graph B: two-stage combine + decoder + heads (GPU) ----
    bIn[bMemSlot].writeFloat(memory)
    bIn[bTsSlot].writeFloat(ts)
    gb.run(bIn, bOut)
    val boxes = bOut[bBoxes].readFloat() // [300*4] cxcywh in [0,1]
    val logits = bOut[bLogits].readFloat() // [300*91]

    // ---- host: decode + per-class NMS ----
    return decode(boxes, logits)
  }

  override fun close() {
    resources.asReversed().forEach { runCatching { it.close() } }
    resources.clear()
  }
}
