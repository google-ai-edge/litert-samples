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

package com.google.ai.edge.examples.model_zoo.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.google.ai.edge.examples.model_zoo.models.liveness.LivenessDetector
import com.google.ai.edge.examples.model_zoo.models.movinet.ActionRecognizer
import com.google.ai.edge.examples.model_zoo.models.sixdrepnet.HeadPose
import com.google.ai.edge.examples.model_zoo.models.sixdrepnet.HeadPoseEstimator
import com.google.ai.edge.examples.model_zoo.models.twinlite.TwinLiteSegmenter
import java.io.File
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

/** Keep an engine alive across camera frames: MoViNet owns its recurrent state until close. */
object RealtimeImageTasks {
  val ids =
    setOf(
      "video-action-recognition",
      "lane-detection",
      "head-pose-estimation",
      "face-liveness-anti-spoofing",
    )

  fun create(
    taskId: String,
    context: Context,
    directory: File,
    backend: String,
  ): SingleImageEngine =
    when (taskId) {
      "video-action-recognition" -> MoViNetImageEngine(context, directory, backend)
      "lane-detection" -> TwinLiteImageEngine(context, directory, backend)
      "head-pose-estimation" -> HeadPoseImageEngine(context, directory, backend)
      "face-liveness-anti-spoofing" -> LivenessImageEngine(context, directory, backend)
      else -> error("Unknown realtime task: $taskId")
    }
}

private fun File.requiredCameraModel(name: String): File =
  File(this, name).also { check(it.isFile) { "Download this task's model first: $name" } }

private fun Bitmap.ownedCopy(): Bitmap = checkNotNull(copy(Bitmap.Config.ARGB_8888, true))

class MoViNetImageEngine(context: Context, directory: File, backend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(backend, "ModelZooMoViNet") {
      ActionRecognizer(context, directory.requiredCameraModel("movinet_a0_stream.tflite"), it)
    }
  private var frames = 0

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val (predictions, ms) = loaded.runner.classify(request.bitmap, 3)
    frames++
    val output = request.bitmap.ownedCopy()
    val canvas = Canvas(output)
    val paint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = output.width / 24f
        setShadowLayer(3f, 0f, 0f, Color.BLACK)
      }
    val text =
      predictions.joinToString("\n") {
        "${it.label}: ${String.format(Locale.US, "%.1f", it.score * 100)}%"
      }
    predictions.forEachIndexed { i, p ->
      canvas.drawText(
        "${i + 1}. ${p.label} ${(p.score * 100).toInt()}%",
        12f,
        (i + 1) * paint.textSize * 1.5f,
        paint,
      )
    }
    return ImageTaskOutput(
      bitmap = output,
      text = text,
      inferenceMs = ms.toDouble(),
      backend = loaded.backend,
      fallbackReason = loaded.fallbackReason,
      backendDetails =
        "GPU compile fails on LiteRT 2.2.0 (RELU_0_TO_1); runs on CPU. Streaming state is retained; the wrapper resets its window every 64 frames.",
      metrics =
        mapOf(
          "topActions" to
            predictions.map {
              mapOf("index" to it.index, "label" to it.label, "score" to it.score)
            },
          "streamFrames" to frames,
          "windowFrames" to 64,
        ),
    )
  }

  override fun close() = loaded.runner.close()
}

class TwinLiteImageEngine(context: Context, directory: File, backend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(backend, "ModelZooTwinLite") {
      TwinLiteSegmenter(context, directory.requiredCameraModel("twinlite.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val (da, ll, ms) = loaded.runner.segment(request.bitmap)
    val W = TwinLiteSegmenter.W
    val H = TwinLiteSegmenter.H
    val ovPixels = IntArray(W * H)
    val GREEN = (0x88 shl 24) or 0x28E05A
    val RED = (0xFF shl 24) or 0xFF3030
    for (i in 0 until W * H) {
      ovPixels[i] =
        when {
          ll[i].toInt() == 1 -> RED
          da[i].toInt() == 1 -> GREEN
          else -> 0
        }
    }
    val ovBitmap = Bitmap.createBitmap(ovPixels, W, H, Bitmap.Config.ARGB_8888)
    val output = request.bitmap.ownedCopy()
    Canvas(output)
      .drawBitmap(
        ovBitmap,
        null,
        Rect(0, 0, output.width, output.height),
        Paint(Paint.FILTER_BITMAP_FLAG),
      )
    ovBitmap.recycle()
    val drivablePixels = da.count { it.toInt() == 1 }
    val lanePixels = ll.count { it.toInt() == 1 }
    return ImageTaskOutput(
      bitmap = output,
      text = "Green: drivable area · Red: lane markings",
      inferenceMs = ms.toDouble(),
      backend = loaded.backend,
      fallbackReason = loaded.fallbackReason,
      metrics =
        mapOf(
          "drivablePixels" to drivablePixels,
          "lanePixels" to lanePixels,
          "maskWidth" to W,
          "maskHeight" to H,
        ),
    )
  }

  override fun close() = loaded.runner.close()
}

class HeadPoseImageEngine(context: Context, directory: File, backend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(backend, "ModelZooHeadPose") {
      HeadPoseEstimator(context, directory.requiredCameraModel("6drepnet.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val bmp = request.bitmap
    val s = minOf(bmp.width, bmp.height)
    val crop = Bitmap.createBitmap(bmp, (bmp.width - s) / 2, (bmp.height - s) / 2, s, s)
    val (pose, ms) = loaded.runner.estimate(crop)
    if (crop !== bmp) crop.recycle()
    val output = bmp.ownedCopy()
    drawHeadPose(Canvas(output), output.width, output.height, pose)
    return ImageTaskOutput(
      bitmap = output,
      text =
        String.format(
          Locale.US,
          "Yaw %.1f° · Pitch %.1f° · Roll %.1f°\nCenter your face in the frame.",
          pose.yaw,
          pose.pitch,
          pose.roll,
        ),
      inferenceMs = ms.toDouble(),
      backend = loaded.backend,
      fallbackReason = loaded.fallbackReason,
      metrics =
        mapOf(
          "yawDegrees" to pose.yaw,
          "pitchDegrees" to pose.pitch,
          "rollDegrees" to pose.roll,
          "crop" to "center square; no face detector",
        ),
    )
  }

  override fun close() = loaded.runner.close()
}

class LivenessImageEngine(context: Context, directory: File, backend: String) : SingleImageEngine {
  private val loaded =
    compileImageBackend(backend, "ModelZooLiveness") {
      LivenessDetector(context, directory.requiredCameraModel("silentface.tflite"), it)
    }

  override fun run(request: ImageTaskRequest): ImageTaskOutput {
    val bmp = request.bitmap
    val s = minOf(bmp.width, bmp.height)
    val crop = Bitmap.createBitmap(bmp, (bmp.width - s) / 2, (bmp.height - s) / 2, s, s)
    val (isLive, score, ms) = loaded.runner.detect(crop)
    if (crop !== bmp) crop.recycle()
    val output = bmp.ownedCopy()
    drawLiveness(Canvas(output), output.width, output.height, isLive, score)
    return ImageTaskOutput(
      bitmap = output,
      text =
        "${if (isLive) "LIVE" else "SPOOF"} · Live score ${String.format(Locale.US, "%.1f", score * 100)}%\nCenter your face in the frame.",
      inferenceMs = ms.toDouble(),
      backend = loaded.backend,
      fallbackReason = loaded.fallbackReason,
      metrics =
        mapOf(
          "isLive" to isLive,
          "liveScore" to score,
          "spoofScore" to (1f - score),
          "crop" to "center square; no face detector",
        ),
    )
  }

  override fun close() = loaded.runner.close()
}

// Render-only drawing math at native bitmap dimensions.
private fun drawHeadPose(canvas: Canvas, width: Int, height: Int, hp: HeadPose) {
  val axis =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      strokeWidth = OverlaySizing.stroke(width)
      style = Paint.Style.STROKE
    }
  val text =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = OverlaySizing.text(width)
      setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }
  val cx = width / 2f
  val cy = height / 2f
  val size = minOf(width, height) * 0.28f
  val p = Math.toRadians(hp.pitch.toDouble())
  val ya = Math.toRadians(-hp.yaw.toDouble())
  val r = Math.toRadians(hp.roll.toDouble())
  val x1 = size * (cos(ya) * cos(r)).toFloat() + cx
  val y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(ya)).toFloat() + cy
  val x2 = size * (-cos(ya) * sin(r)).toFloat() + cx
  val y2 = size * (cos(p) * cos(r) - sin(p) * sin(ya) * sin(r)).toFloat() + cy
  val x3 = size * sin(ya).toFloat() + cx
  val y3 = size * (-cos(ya) * sin(p)).toFloat() + cy
  axis.color = Color.rgb(255, 60, 60)
  canvas.drawLine(cx, cy, x1, y1, axis)
  axis.color = Color.rgb(60, 220, 90)
  canvas.drawLine(cx, cy, x2, y2, axis)
  axis.color = Color.rgb(70, 130, 255)
  canvas.drawLine(cx, cy, x3, y3, axis)
  canvas.drawText(
    "yaw ${hp.yaw.toInt()}  pitch ${hp.pitch.toInt()}  roll ${hp.roll.toInt()}",
    12f * OverlaySizing.unit(width),
    height - 20f * OverlaySizing.unit(width),
    text,
  )
}

private fun drawLiveness(canvas: Canvas, width: Int, height: Int, isLive: Boolean, score: Float) {
  val box =
    Paint().apply {
      style = Paint.Style.STROKE
      strokeWidth = OverlaySizing.stroke(width)
    }
  val text =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      textSize = 24f * OverlaySizing.unit(width)
      setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }
  val sub =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = OverlaySizing.text(width)
      setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }
  val col = if (isLive) Color.rgb(50, 220, 100) else Color.rgb(240, 70, 70)
  val s = minOf(width, height) * 0.6f
  val l = (width - s) / 2
  val tp = (height - s) / 2
  box.color = col
  canvas.drawRect(l, tp, l + s, tp + s, box)
  text.color = col
  val label = if (isLive) "LIVE" else "SPOOF"
  canvas.drawText(label, l, tp - 10f * OverlaySizing.unit(width), text)
  canvas.drawText(
    "live score ${(score * 100).toInt()}%",
    l,
    tp + s + 22f * OverlaySizing.unit(width),
    sub,
  )
}
