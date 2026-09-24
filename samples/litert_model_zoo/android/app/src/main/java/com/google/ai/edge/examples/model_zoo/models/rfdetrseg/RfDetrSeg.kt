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

package com.google.ai.edge.examples.model_zoo.models.rfdetrseg

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * RF-DETR-Seg Nano instance segmentation, fully on the LiteRT CompiledModel GPU (ML Drift).
 *
 * RF-DETR-Seg is a two-stage DETR (DINOv2-S/12 backbone + deformable decoder + mask head) whose
 * query selection (topk + gather) has no GPU-compatible op, so the model is split into two GPU
 * graphs with a tiny host step between them:
 *
 * Graph A (GPU) image[1,3,312,312] + clspos[1,1,384] + pospatch[1,676,384] -> enc_class[1,676,91],
 * enc_delta[1,676,4], memory*2[1,676,256] host (here) memory/2 -> proposal-grid combine -> topk-100
 * by max class score -> gather coords -> two-stage reparam with the learned refpoint_embed Graph B
 * (GPU) (memory, refpoint[1,100,4], query_feat[1,100,256]) -> boxes[1,100,4] (cxcywh),
 * logits[1,100,91], masks[1,100,78,78] host (here) sigmoid + threshold + per-class NMS ->
 * detections with full-image masks
 *
 * ML Drift mis-executes compute chains that consume LARGE BAKED CONSTANTS (fp32 gives the same
 * wrong numbers — not a precision issue), so the cls+pos embedding, the decoder query embedding and
 * the refpoint reparam constants are all HOST-FED from assets instead of living in the graph;
 * memory is emitted as memory*2 because a [1,N,C] tensor that is both consumed and output comes
 * back zeroed. Both graphs run 100% on the GPU (1293/1293 and 884/884 nodes LITERT_CL on a Pixel
 * 8a, 17.5 ms + 9.1 ms).
 */
class RfDetrSeg(
  private val ctx: Context,
  private val modelDir: File,
  private val accelerator: Accelerator = Accelerator.GPU,
) : Closeable {

  companion object {
    const val SIZE = 312
    const val MODEL_A = "rfdetrseg_graphA_fp16.tflite"
    const val MODEL_B = "rfdetrseg_graphB_fp16.tflite"
    const val GRID = 26 // 26x26 single deformable level
    const val NPROP = GRID * GRID // 676 proposals
    const val NQ = 100 // decoder queries
    const val NCLS = 91 // COCO id space (index == COCO category id)
    const val HID = 256
    const val MASK = 78 // 78x78 full-image mask grid (312 / 4)
    const val PROP_WH = 0.05f // proposal-grid prior box size
    // ImageNet normalization (RF-DETR preprocessing)
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    const val SCORE_THRESH = 0.5f
    const val IOU_THRESH = 0.6f // light NMS — cleans fp16 near-duplicate queries

    internal fun nms(dets: List<Detection>): List<Detection> {
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

    internal fun iou(a: Detection, b: Detection): Float {
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

  /**
   * Box coords are normalized [0,1] in squashed-SIZE space; mask is the query's raw-logit
   * full-image 78x78 grid (inside = logit > 0).
   */
  data class Detection(
    val cls: Int,
    val score: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val mask: FloatArray,
  )

  private val closeActions = mutableListOf<() -> Unit>()
  private val env =
    Environment.create().also {
      closeActions.add { it.close() }
    } // ONE shared env — a null env leaks an OpenCL context per create

  private fun load(name: String): CompiledModel =
    try {
      val f = File(modelDir, name)
      check(f.exists()) { "Model not downloaded: $name" }
      CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), env).also {
        closeActions.add { it.close() }
      }
    } catch (failure: Exception) {
      close()
      throw failure
    }

  private fun buffers(model: CompiledModel, input: Boolean) =
    try {
      (if (input) model.createInputBuffers() else model.createOutputBuffers()).also { values ->
        values.forEach { buffer -> closeActions.add { buffer.close() } }
      }
    } catch (failure: Exception) {
      close()
      throw failure
    }

  /** Raw little-endian float32 asset (exported from the conversion script's host_*.npy). */
  private fun loadAsset(name: String, n: Int): FloatArray =
    try {
      val bytes = File(modelDir, name).readBytes()
      check(bytes.size == n * 4) { "$name: ${bytes.size} bytes, expected ${n * 4}" }
      val out = FloatArray(n)
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
      out
    } catch (failure: Exception) {
      close()
      throw failure
    }

  private val refpointEmbed = loadAsset("refpoint_embed.bin", NQ * 4)

  private val ga = load(MODEL_A)
  private val gb = load(MODEL_B)
  private val aIn = buffers(ga, true)
  private val aOut = buffers(ga, false)
  private val bIn = buffers(gb, true)
  private val bOut = buffers(gb, false)

  // Resolve buffer slots by float capacity (robust to converter ordering).
  private val aImage = aIn.indexOfFirst { it.readFloat().size == 3 * SIZE * SIZE }
  private val aClspos = aIn.indexOfFirst { it.readFloat().size == 384 }
  private val aPospatch = aIn.indexOfFirst { it.readFloat().size == NPROP * 384 }
  private val aEncClass = aOut.indexOfFirst { it.readFloat().size == NPROP * NCLS }
  private val aEncDelta = aOut.indexOfFirst { it.readFloat().size == NPROP * 4 }
  private val aMemory = aOut.indexOfFirst { it.readFloat().size == NPROP * HID }
  private val bMemSlot = bIn.indexOfFirst { it.readFloat().size == NPROP * HID }
  private val bRefSlot = bIn.indexOfFirst { it.readFloat().size == NQ * 4 }
  private val bQfSlot = bIn.indexOfFirst { it.readFloat().size == NQ * HID }
  private val bBoxes = bOut.indexOfFirst { it.readFloat().size == NQ * 4 }
  private val bLogits = bOut.indexOfFirst { it.readFloat().size == NQ * NCLS }
  private val bMasks = bOut.indexOfFirst { it.readFloat().size == NQ * MASK * MASK }

  init {
    try {
      check(
        listOf(
            aImage,
            aClspos,
            aPospatch,
            aEncClass,
            aEncDelta,
            aMemory,
            bMemSlot,
            bRefSlot,
            bQfSlot,
            bBoxes,
            bLogits,
            bMasks,
          )
          .all { it >= 0 }
      ) {
        "RF-DETR-Seg tensor shape mismatch"
      }
      // The host-fed constants never change — write them once.
      aIn[aClspos].writeFloat(loadAsset("clspos.bin", 384))
      aIn[aPospatch].writeFloat(loadAsset("pospatch.bin", NPROP * 384))
      bIn[bQfSlot].writeFloat(loadAsset("query_feat.bin", NQ * HID))
    } catch (failure: Exception) {
      close()
      throw failure
    }
  }

  /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns detections with normalized boxes + 78x78 masks. */
  fun detect(rgb: FloatArray): List<Detection> {
    // ---- Graph A: backbone + encoder + proposal heads (GPU) ----
    val chw = FloatArray(3 * SIZE * SIZE)
    val hw = SIZE * SIZE
    for (i in 0 until hw) {
      chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
      chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
      chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
    }
    aIn[aImage].writeFloat(chw)
    ga.run(aIn, aOut)
    val encClass = aOut[aEncClass].readFloat() // [676*91]
    val encDelta = aOut[aEncDelta].readFloat() // [676*4]
    val memory = aOut[aMemory].readFloat() // [676*256], x2 on the graph side
    for (i in memory.indices) memory[i] *= 0.5f // invert the output-copy trick

    // ---- host: proposal-grid combine -> topk-100 -> gather -> two-stage reparam ----
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
    val refpoint = FloatArray(NQ * 4)
    for (i in 0 until NQ) {
      val p = order[i]
      // proposal grid: cxcy = (grid + 0.5) / GRID, wh = PROP_WH (image-independent)
      val pcx = (p % GRID + 0.5f) / GRID
      val pcy = (p / GRID + 0.5f) / GRID
      val d = p * 4
      val tcx = encDelta[d] * PROP_WH + pcx
      val tcy = encDelta[d + 1] * PROP_WH + pcy
      val tw = Math.exp(encDelta[d + 2].toDouble()).toFloat() * PROP_WH
      val th = Math.exp(encDelta[d + 3].toDouble()).toFloat() * PROP_WH
      val r = i * 4
      refpoint[r] = refpointEmbed[r] * tw + tcx
      refpoint[r + 1] = refpointEmbed[r + 1] * th + tcy
      refpoint[r + 2] = Math.exp(refpointEmbed[r + 2].toDouble()).toFloat() * tw
      refpoint[r + 3] = Math.exp(refpointEmbed[r + 3].toDouble()).toFloat() * th
    }

    // ---- Graph B: decoder + box/class heads + mask head (GPU) ----
    bIn[bMemSlot].writeFloat(memory)
    bIn[bRefSlot].writeFloat(refpoint)
    gb.run(bIn, bOut)
    val boxes = bOut[bBoxes].readFloat() // [100*4] cxcywh in [0,1]
    val logits = bOut[bLogits].readFloat() // [100*91]
    val masks = bOut[bMasks].readFloat() // [100*78*78] raw logits

    // ---- host: decode + per-class NMS ----
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
      val mask = masks.copyOfRange(q * MASK * MASK, (q + 1) * MASK * MASK)
      dets.add(
        Detection(
          bestC,
          score,
          boxes[q * 4],
          boxes[q * 4 + 1],
          boxes[q * 4 + 2],
          boxes[q * 4 + 3],
          mask,
        )
      )
    }
    return nms(dets)
  }

  override fun close() {
    closeActions.asReversed().forEach { runCatching { it() } }
    closeActions.clear()
  }
}
