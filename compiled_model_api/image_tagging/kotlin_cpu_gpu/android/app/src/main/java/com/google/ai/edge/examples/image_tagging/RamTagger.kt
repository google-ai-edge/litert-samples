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

package com.google.ai.edge.examples.image_tagging

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * RAM++ multi-label image tagging on the LiteRT CompiledModel API, device-verified hybrid pipeline:
 *   G1  Swin stages 0-2   image[1,3,384,384] -> feat[1,144,1536]     (GPU)
 *   C2  Swin stage 3 tail feat -> image_embeds[1,145,512]            (CPU; stage-3 fp16-accum wall)
 *   R   reweight          cls[1,512] -> queries[1,4585,768]          (CPU; 479MB frozen tag bank)
 *   B   Query2Label head  queries + image_embeds -> logits[1,4585]   (GPU)
 * then host sigmoid + per-class threshold + tag lookup.
 */
class RamTagger(private val ctx: Context) {

  companion object {
    const val SIZE = 384
    const val NCLASS = 4585
    const val TDIM = 512
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private fun f(n: String) = File(ctx.filesDir, n).also {
    check(it.exists()) { "Missing ${it.name}. Run install_to_device.sh first." }
  }

  private val g1 = CompiledModel.create(f("ram_swin_s012_fp16.tflite").absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  private val g1In = g1.createInputBuffers()
  private val g1Out = g1.createOutputBuffers()
  private val c2 = CompiledModel.create(f("ram_stage3_tail_fp16.tflite").absolutePath, CompiledModel.Options(Accelerator.CPU), null)
  private val c2In = c2.createInputBuffers()
  private val c2Out = c2.createOutputBuffers()
  private val rw = CompiledModel.create(f("ram_reweight_fp16.tflite").absolutePath, CompiledModel.Options(Accelerator.CPU), null)
  private val rwIn = rw.createInputBuffers()
  private val rwOut = rw.createOutputBuffers()
  private val th = CompiledModel.create(f("ram_taghead_fp16.tflite").absolutePath, CompiledModel.Options(Accelerator.GPU), null)
  private val thIn = th.createInputBuffers()
  private val thOut = th.createOutputBuffers()

  private val tags = ctx.assets.open("ram_tag_list.txt").bufferedReader().readLines()
  private val thresh = ctx.assets.open("ram_tag_threshold.bin").readBytes().let {
    val bb = ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN)
    FloatArray(it.size / 4) { bb.float } }

  private fun preprocess(bm: Bitmap): FloatArray {
    val s = Bitmap.createScaledBitmap(bm, SIZE, SIZE, true)
    val px = IntArray(SIZE * SIZE)
    s.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
    val out = FloatArray(3 * SIZE * SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = px[i]
      out[i] = (((p shr 16 and 0xFF) / 255f) - MEAN[0]) / STD[0]
      out[plane + i] = (((p shr 8 and 0xFF) / 255f) - MEAN[1]) / STD[1]
      out[2 * plane + i] = (((p and 0xFF) / 255f) - MEAN[2]) / STD[2]
    }
    return out
  }

  data class Tag(val name: String, val prob: Float)

  fun tag(bm: Bitmap, topK: Int = 40): List<Tag> {
    g1In[0].writeFloat(preprocess(bm))
    g1.run(g1In, g1Out)
    val feat = g1Out[0].readFloat()

    c2In[0].writeFloat(feat)
    c2.run(c2In, c2Out)
    val iemb = c2Out[0].readFloat()
    val cls = iemb.copyOfRange(0, TDIM)

    rwIn[0].writeFloat(cls)
    rw.run(rwIn, rwOut)
    val queries = rwOut[0].readFloat()

    for (b in thIn) {
        val n = b.readFloat().size
        b.writeFloat(if (n == queries.size) queries else iemb)
    }
    th.run(thIn, thOut)
    val logits = thOut[0].readFloat()

    val hits = ArrayList<Tag>()
    for (i in 0 until NCLASS) {
      val p = 1f / (1f + exp(-logits[i]))
      if (p > thresh[i]) {
          hits.add(Tag(tags[i], p))
      }
    }
    hits.sortByDescending { it.prob }
    return if (hits.size > topK) hits.subList(0, topK) else hits
  }

  fun close() {
      g1.close()
      c2.close()
      rw.close()
      th.close()
  }
}
