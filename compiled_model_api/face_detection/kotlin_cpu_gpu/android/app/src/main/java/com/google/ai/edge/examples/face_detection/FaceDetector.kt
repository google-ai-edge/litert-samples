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

package com.google.ai.edge.examples.face_detection

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YuNet (ShiqiYu/libfacedetection) face detection on the LiteRT CompiledModel GPU.
 *   image[1,3,640,640] (BGR, 0-255, no normalization) -> 12 anchor-free outputs (cls/obj/bbox/kps × 3 strides)
 *
 * Tiny (0.076 M params, 0.3 MB fp16). Decodes faces + 5 landmarks host-side: score = cls·obj, box =
 * center + exp(wh), then NMS. ~4 ms / 640x640 on a Pixel 8a, fully GPU. Output order is
 * [cls8,cls16,cls32, obj8,obj16,obj32, bbox8,bbox16,bbox32, kps8,kps16,kps32]; cls/obj are sigmoid-baked.
 */
class FaceDetector(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val SIZE = 640
        val STRIDES = intArrayOf(8, 16, 32)
        const val SCORE_THRESH = 0.6f
        const val NMS_THRESH = 0.45f
    }

    /** Box in SIZE×SIZE space (x1,y1,x2,y2), score, and 5 (x,y) landmarks. */
    data class Face(val x1: Float, val y1: Float, val x2: Float, val y2: Float,
                    val score: Float, val landmarks: FloatArray)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "yunet_fp16.tflite")
        check(f.exists()) { "Model not found: yunet_fp16.tflite. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns detected faces (in SIZE×SIZE space). */
    fun detect(rgb: FloatArray): List<Face> {
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {                     // RGB -> BGR planar, raw 0-255
            chw[i] = rgb[i * 3 + 2]
            chw[hw + i] = rgb[i * 3 + 1]
            chw[2 * hw + i] = rgb[i * 3]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val cls = Array(3) { outBuf[it].readFloat() }
        val obj = Array(3) { outBuf[3 + it].readFloat() }
        val bbox = Array(3) { outBuf[6 + it].readFloat() }
        val kps = Array(3) { outBuf[9 + it].readFloat() }

        val faces = ArrayList<Face>()
        for (li in 0 until 3) {
            val s = STRIDES[li]; val fw = SIZE / s
            for (i in 0 until fw * fw) {
                val score = cls[li][i] * obj[li][i]
                if (score < SCORE_THRESH) continue
                val col = i % fw; val row = i / fw; val px = (col * s).toFloat(); val py = (row * s).toFloat()
                val b = i * 4
                val cx = bbox[li][b] * s + px; val cy = bbox[li][b + 1] * s + py
                val w = exp(bbox[li][b + 2]) * s; val h = exp(bbox[li][b + 3]) * s
                val k = i * 10
                val lm = FloatArray(10)
                for (j in 0 until 5) { lm[2 * j] = kps[li][k + 2 * j] * s + px; lm[2 * j + 1] = kps[li][k + 2 * j + 1] * s + py }
                faces.add(Face(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, lm))
            }
        }
        return nms(faces)
    }

    private fun nms(faces: List<Face>): List<Face> {
        val sorted = faces.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Face>()
        while (sorted.isNotEmpty()) {
            val f = sorted.removeAt(0); keep.add(f)
            sorted.removeAll { iou(f, it) >= NMS_THRESH }
        }
        return keep
    }

    private fun iou(a: Face, b: Face): Float {
        val x1 = max(a.x1, b.x1); val y1 = max(a.y1, b.y1); val x2 = min(a.x2, b.x2); val y2 = min(a.y2, b.y2)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (ua > 0f) inter / ua else 0f
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
