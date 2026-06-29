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

package com.google.ai.edge.examples.rt_detr_v2

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * RT-DETRv2-S object detection, fully on the LiteRT CompiledModel GPU (ML Drift / LITERT_CL).
 *
 * RT-DETRv2 is a two-stage DETR whose query selection (topk + gather) has no GPU-compatible op, so the
 * model is split into two GPU graphs with a host step between them:
 *
 *   Graph A (GPU)  image[1,3,640,640] -> enc_class[1,8400,80], memory_raw*2[1,8400,256]
 *   host (here)    topk-300 by max class score; for the 300 selected tokens compute, in fp32,
 *                  target = enc_output(valid*memory_raw)   (Linear + LayerNorm, per-token)
 *                  ref    = enc_bbox_head(target) + anchors (3-layer MLP, per-token)
 *   Graph B (GPU)  (memory_raw, target, ref) -> boxes[1,300,4] (cxcywh), logits[1,300,80]
 *   host (here)    sigmoid + score threshold + cxcywh->xyxy + light NMS -> detections
 *
 * Why the per-token tail (enc_output + bbox_head) is on the host: on the Mali fp16 path a 3D *token*
 * tensor [1,N,256] that fans out to several consumers inside one GPU graph gets silently corrupted
 * (the longer branch is clobbered) — enc_bbox_head's reference boxes collapsed to the default anchor
 * and large objects were lost. enc_output/enc_bbox_head are per-token (no cross-token mixing), so
 * gather(f(x),idx) == f(gather(x,idx)); running them on the host over only the 300 selected tokens is
 * exact and cheap. Graph A then emits only the two fp16-clean tensors (enc_class + memory_raw). The
 * x2 on memory_raw forces a separate, clean output buffer (exact in fp16; divided back here).
 *
 * Both graphs run 100% on the GPU on a Pixel 8a (Graph A ~6 ms, Graph B ~11 ms). Device output matches
 * PyTorch (COCO val giraffe 7/7, cats 6/6, IoU 0.98-1.00).
 */
class RtDetr(private val ctx: Context) : Closeable {

  companion object {
    const val SIZE = 640
    const val MODEL_A = "rtdetr_graphA_fp16.tflite"
    const val MODEL_B = "rtdetr_graphB_fp16.tflite"
    const val NPROP = 8400         // 80*80 + 40*40 + 20*20
    const val NQ = 300             // decoder queries
    const val NCLS = 80            // COCO contiguous 0..79 (matches coco_labels.txt order)
    const val HID = 256
    const val MEM_SCALE = 2f       // Graph A emits memory_raw * MEM_SCALE; undo here
    const val LN_EPS = 1e-5f
    // RT-DETR preprocessing = rescale to [0,1], NO ImageNet mean/std (do_normalize=False).
    const val SCORE_THRESH = 0.4f
    const val IOU_THRESH = 0.7f    // light NMS — RT-DETR is NMS-free by design; cleans fp16 near-dupes
  }

  /** Box coords are normalized [0,1] in the squashed SIZE×SIZE space. */
  data class Detection(val cls: Int, val score: Float, val cx: Float, val cy: Float, val w: Float, val h: Float)

  private fun load(name: String): CompiledModel {
    val f = File(ctx.filesDir, name)
    check(f.exists()) { "Model not found: $name. Push first: install_to_device.sh" }
    return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  }

  private val ga = load(MODEL_A)
  private val gb = load(MODEL_B)
  private val aIn = ga.createInputBuffers()
  private val aOut = ga.createOutputBuffers()
  private val bIn = gb.createInputBuffers()
  private val bOut = gb.createOutputBuffers()

  // Resolve buffer slots by float capacity (robust to converter ordering).
  private val aEncClass = aOut.indexOfFirst { it.readFloat().size == NPROP * NCLS }
  private val aMemory = aOut.indexOfFirst { it.readFloat().size == NPROP * HID }
  private val bMemSlot = bIn.indexOfFirst { it.readFloat().size == NPROP * HID }
  private val bTgtSlot = bIn.indexOfFirst { it.readFloat().size == NQ * HID }
  private val bRefSlot = bIn.indexOfFirst { it.readFloat().size == NQ * 4 }
  private val bBoxes = bOut.indexOfFirst { it.readFloat().size == NQ * 4 }
  private val bLogits = bOut.indexOfFirst { it.readFloat().size == NQ * NCLS }

  // ---- host-side per-token tail weights (assets/host_params.bin, layout fixed by the build script) ----
  private val eoW: FloatArray; private val eoB: FloatArray; private val eoG: FloatArray; private val eoBeta: FloatArray
  private val bb0W: FloatArray; private val bb0B: FloatArray
  private val bb1W: FloatArray; private val bb1B: FloatArray
  private val bb2W: FloatArray; private val bb2B: FloatArray
  private val valid: FloatArray; private val anchors: FloatArray

  init {
    val raw = ctx.assets.open("host_params.bin").readBytes()
    val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    fun take(n: Int) = FloatArray(n).also { fb.get(it) }
    eoW = take(HID * HID); eoB = take(HID); eoG = take(HID); eoBeta = take(HID)
    bb0W = take(HID * HID); bb0B = take(HID)
    bb1W = take(HID * HID); bb1B = take(HID)
    bb2W = take(4 * HID); bb2B = take(4)
    valid = take(NPROP); anchors = take(NPROP * 4)
  }

  /** y[o] = sum_i x[i]*W[o*inDim+i] + b[o] ; optional ReLU. */
  private fun linear(x: FloatArray, w: FloatArray, b: FloatArray, outDim: Int, inDim: Int, relu: Boolean): FloatArray {
    val y = FloatArray(outDim)
    for (o in 0 until outDim) {
      var s = b[o]; val base = o * inDim
      for (i in 0 until inDim) s += x[i] * w[base + i]
      y[o] = if (relu && s < 0f) 0f else s
    }
    return y
  }

  /** enc_output = Linear(256->256) then LayerNorm(256). */
  private fun encOutput(memSel: FloatArray): FloatArray {
    val y = linear(memSel, eoW, eoB, HID, HID, false)
    var mean = 0f; for (v in y) mean += v; mean /= HID
    var varr = 0f; for (v in y) { val d = v - mean; varr += d * d }; varr /= HID
    val inv = (1.0 / Math.sqrt((varr + LN_EPS).toDouble())).toFloat()
    val out = FloatArray(HID)
    for (o in 0 until HID) out[o] = (y[o] - mean) * inv * eoG[o] + eoBeta[o]
    return out
  }

  /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns detections (boxes normalized in [0,1] SIZE space). */
  fun detect(rgb: FloatArray): List<Detection> {
    // ---- Graph A (GPU): backbone + hybrid encoder + score head ----
    val chw = FloatArray(3 * SIZE * SIZE)
    val hw = SIZE * SIZE
    for (i in 0 until hw) {                       // CHW, [0,1] rescale only
      chw[i] = rgb[i * 3] / 255f
      chw[hw + i] = rgb[i * 3 + 1] / 255f
      chw[2 * hw + i] = rgb[i * 3 + 2] / 255f
    }
    aIn[0].writeFloat(chw)
    ga.run(aIn, aOut)
    val encClass = aOut[aEncClass].readFloat()    // [8400*80]
    val memScaled = aOut[aMemory].readFloat()      // [8400*256] = memory_raw * MEM_SCALE

    // ---- host: topk-300 by max class score (descending, matches torch.topk) ----
    val maxScore = FloatArray(NPROP)
    for (p in 0 until NPROP) {
      var m = -Float.MAX_VALUE; val base = p * NCLS
      for (c in 0 until NCLS) { val v = encClass[base + c]; if (v > m) m = v }
      maxScore[p] = m
    }
    val order = (0 until NPROP).sortedByDescending { maxScore[it] }

    // ---- host: per-token tail (fp32) on the 300 selected; build GraphB target + ref ----
    val invScale = 1f / MEM_SCALE
    val memRaw = FloatArray(NPROP * HID)           // raw source_flatten for GraphB cross-attn
    for (i in memRaw.indices) memRaw[i] = memScaled[i] * invScale
    val target = FloatArray(NQ * HID)
    val ref = FloatArray(NQ * 4)
    val memSel = FloatArray(HID)
    for (q in 0 until NQ) {
      val p = order[q]; val mb = p * HID; val vp = valid[p]
      for (c in 0 until HID) memSel[c] = vp * memRaw[mb + c]     // masked memory
      val tgt = encOutput(memSel)
      System.arraycopy(tgt, 0, target, q * HID, HID)
      val delta = run {                                          // bbox_head MLP
        val h0 = linear(tgt, bb0W, bb0B, HID, HID, true)
        val h1 = linear(h0, bb1W, bb1B, HID, HID, true)
        linear(h1, bb2W, bb2B, 4, HID, false)
      }
      val ab = p * 4
      for (k in 0 until 4) ref[q * 4 + k] = delta[k] + anchors[ab + k]
    }

    // ---- Graph B (GPU): two-stage combine + plain decoder + heads ----
    bIn[bMemSlot].writeFloat(memRaw)
    bIn[bTgtSlot].writeFloat(target)
    bIn[bRefSlot].writeFloat(ref)
    gb.run(bIn, bOut)
    val boxes = bOut[bBoxes].readFloat()           // [300*4] cxcywh in [0,1]
    val logits = bOut[bLogits].readFloat()         // [300*80]

    // ---- host: decode + per-class NMS ----
    val dets = ArrayList<Detection>()
    for (q in 0 until NQ) {
      var best = -Float.MAX_VALUE; var bestC = -1
      val base = q * NCLS
      for (c in 0 until NCLS) { val v = logits[base + c]; if (v > best) { best = v; bestC = c } }
      val score = 1f / (1f + Math.exp(-best.toDouble()).toFloat())  // sigmoid
      if (score < SCORE_THRESH) continue
      dets.add(Detection(bestC, score, boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]))
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
        for (j in i + 1 until group.size) if (!taken[j] && iou(group[i], group[j]) > IOU_THRESH) taken[j] = true
      }
    }
    return out.sortedByDescending { it.score }
  }

  private fun iou(a: Detection, b: Detection): Float {
    val ax0 = a.cx - a.w / 2; val ay0 = a.cy - a.h / 2; val ax1 = a.cx + a.w / 2; val ay1 = a.cy + a.h / 2
    val bx0 = b.cx - b.w / 2; val by0 = b.cy - b.h / 2; val bx1 = b.cx + b.w / 2; val by1 = b.cy + b.h / 2
    val ix0 = maxOf(ax0, bx0); val iy0 = maxOf(ay0, by0); val ix1 = minOf(ax1, bx1); val iy1 = minOf(ay1, by1)
    val iw = maxOf(0f, ix1 - ix0); val ih = maxOf(0f, iy1 - iy0); val inter = iw * ih
    val ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return if (ua > 0f) inter / ua else 0f
  }

  override fun close() {
    aIn.forEach { it.close() }; aOut.forEach { it.close() }; ga.close()
    bIn.forEach { it.close() }; bOut.forEach { it.close() }; gb.close()
  }
}
