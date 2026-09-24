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

package com.google.ai.edge.examples.model_zoo.view

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathBuilder
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp

/** Original geometric vector artwork drawn for this app; no third-party logo or icon source. */
internal fun taskIcon(taskId: String): ImageVector = TaskIcons[taskId] ?: FallbackIcon

private fun glyph(name: String, draw: PathBuilder.() -> Unit): ImageVector =
  ImageVector.Builder(
      name = "ModelZoo.$name",
      defaultWidth = 24.dp,
      defaultHeight = 24.dp,
      viewportWidth = 24f,
      viewportHeight = 24f,
    )
    .apply {
      path(
        fill = null,
        stroke = SolidColor(Color.Black),
        strokeLineWidth = 1.6f,
        strokeLineCap = StrokeCap.Round,
        strokeLineJoin = StrokeJoin.Round,
        pathBuilder = draw,
      )
    }
    .build()

private fun PathBuilder.line(x1: Float, y1: Float, x2: Float, y2: Float) {
  moveTo(x1, y1)
  lineTo(x2, y2)
}

private fun PathBuilder.box(x: Float, y: Float, width: Float, height: Float) {
  moveTo(x, y)
  lineTo(x + width, y)
  lineTo(x + width, y + height)
  lineTo(x, y + height)
  close()
}

private fun PathBuilder.circle(x: Float, y: Float, radius: Float) {
  val c = radius * 0.55228475f
  moveTo(x + radius, y)
  curveTo(x + radius, y + c, x + c, y + radius, x, y + radius)
  curveTo(x - c, y + radius, x - radius, y + c, x - radius, y)
  curveTo(x - radius, y - c, x - c, y - radius, x, y - radius)
  curveTo(x + c, y - radius, x + radius, y - c, x + radius, y)
  close()
}

private fun PathBuilder.wave(y: Float) {
  moveTo(3f, y)
  curveTo(6f, y - 6f, 8f, y - 6f, 11f, y)
  curveTo(14f, y + 6f, 17f, y + 6f, 21f, y)
}

private val FallbackIcon =
  glyph("model") {
    box(4f, 4f, 6f, 6f)
    box(14f, 4f, 6f, 6f)
    box(4f, 14f, 6f, 6f)
    box(14f, 14f, 6f, 6f)
  }

private val TaskIcons by lazy {
  mapOf(
    "object-detection" to
      glyph("detection") {
        moveTo(3f, 8f)
        lineTo(3f, 3f)
        lineTo(8f, 3f)
        moveTo(16f, 3f)
        lineTo(21f, 3f)
        lineTo(21f, 8f)
        moveTo(21f, 16f)
        lineTo(21f, 21f)
        lineTo(16f, 21f)
        moveTo(8f, 21f)
        lineTo(3f, 21f)
        lineTo(3f, 16f)
        box(7f, 7f, 10f, 10f)
      },
    "video-action-recognition" to
      glyph("video") {
        box(3f, 5f, 18f, 14f)
        moveTo(10f, 8f)
        lineTo(16f, 12f)
        lineTo(10f, 16f)
        close()
        line(3f, 9f, 6f, 9f)
        line(3f, 15f, 6f, 15f)
      },
    "semantic-segmentation" to
      glyph("semantic") {
        box(3f, 3f, 18f, 18f)
        moveTo(3f, 12f)
        lineTo(9f, 8f)
        lineTo(15f, 13f)
        lineTo(21f, 9f)
        moveTo(3f, 18f)
        lineTo(8f, 14f)
        lineTo(15f, 19f)
        lineTo(21f, 15f)
      },
    "lane-detection" to
      glyph("lane") {
        moveTo(3f, 21f)
        curveTo(7f, 15f, 8f, 9f, 9f, 3f)
        moveTo(21f, 21f)
        curveTo(17f, 15f, 16f, 9f, 15f, 3f)
        line(12f, 4f, 12f, 7f)
        line(12f, 11f, 12f, 14f)
        line(12f, 18f, 12f, 21f)
      },
    "super-resolution" to
      glyph("edsr") {
        box(3f, 13f, 8f, 8f)
        line(8f, 16f, 20f, 4f)
        moveTo(13f, 4f)
        lineTo(20f, 4f)
        lineTo(20f, 11f)
        line(3f, 9f, 3f, 3f)
        line(3f, 3f, 9f, 3f)
      },
    "super-resolution-real-esrgan" to
      glyph("esrgan") {
        box(3f, 3f, 18f, 18f)
        box(6f, 6f, 5f, 5f)
        moveTo(12f, 18f)
        lineTo(18f, 12f)
        lineTo(18f, 17f)
        line(18f, 12f, 13f, 12f)
      },
    "image-dehazing" to
      glyph("dehaze") {
        circle(12f, 7f, 3f)
        line(12f, 1f, 12f, 2f)
        line(6f, 3f, 7f, 4f)
        line(18f, 3f, 17f, 4f)
        line(3f, 13f, 21f, 13f)
        line(5f, 17f, 19f, 17f)
        line(3f, 21f, 21f, 21f)
      },
    "face-liveness-anti-spoofing" to
      glyph("liveness") {
        moveTo(12f, 2f)
        lineTo(20f, 6f)
        lineTo(19f, 15f)
        curveTo(17f, 19f, 15f, 21f, 12f, 22f)
        curveTo(9f, 21f, 7f, 19f, 5f, 15f)
        lineTo(4f, 6f)
        close()
        circle(12f, 10f, 3f)
        moveTo(8f, 17f)
        curveTo(9f, 14f, 15f, 14f, 16f, 17f)
      },
    "head-pose-estimation" to
      glyph("pose") {
        circle(11f, 10f, 6f)
        line(11f, 10f, 21f, 10f)
        line(11f, 10f, 11f, 21f)
        line(11f, 10f, 3f, 18f)
        line(19f, 8f, 21f, 10f)
        line(21f, 10f, 19f, 12f)
      },
    "crowd-counting" to
      glyph("crowd") {
        circle(5f, 7f, 2f)
        circle(12f, 6f, 2.5f)
        circle(19f, 7f, 2f)
        moveTo(2f, 17f)
        lineTo(2f, 14f)
        curveTo(2f, 10f, 8f, 10f, 8f, 14f)
        moveTo(8f, 20f)
        lineTo(8f, 14f)
        curveTo(8f, 9f, 16f, 9f, 16f, 14f)
        lineTo(16f, 20f)
        moveTo(16f, 14f)
        curveTo(16f, 10f, 22f, 10f, 22f, 14f)
        lineTo(22f, 17f)
      },
    "instance-segmentation" to
      glyph("instance") {
        box(3f, 4f, 11f, 12f)
        box(10f, 10f, 11f, 11f)
        line(5f, 8f, 9f, 8f)
        line(13f, 14f, 17f, 14f)
        line(13f, 17f, 18f, 17f)
      },
    "background-removal" to
      glyph("cutout") {
        moveTo(10f, 3f)
        curveTo(5f, 3f, 3f, 7f, 5f, 11f)
        curveTo(1f, 16f, 4f, 21f, 9f, 21f)
        lineTo(14f, 21f)
        curveTo(20f, 20f, 20f, 16f, 17f, 13f)
        curveTo(21f, 8f, 19f, 4f, 14f, 3f)
        close()
        line(3f, 3f, 6f, 3f)
        line(21f, 3f, 21f, 6f)
        line(3f, 18f, 3f, 21f)
        line(18f, 21f, 21f, 21f)
      },
    "portrait-matting" to
      glyph("matte") {
        box(3f, 2f, 18f, 20f)
        circle(12f, 9f, 3.5f)
        moveTo(6f, 21f)
        curveTo(6f, 13f, 18f, 13f, 18f, 21f)
        line(18f, 4f, 20f, 6f)
        line(18f, 8f, 20f, 10f)
      },
    "dense-feature-visualization" to
      glyph("features") {
        circle(5f, 5f, 2f)
        circle(12f, 5f, 1f)
        circle(19f, 5f, 2f)
        circle(5f, 12f, 1f)
        circle(12f, 12f, 3f)
        circle(19f, 12f, 1f)
        circle(5f, 19f, 2f)
        circle(12f, 19f, 1f)
        circle(19f, 19f, 2f)
      },
    "speech-recognition" to
      glyph("speech-input") {
        moveTo(5f, 9f)
        lineTo(5f, 7f)
        curveTo(5f, 2f, 11f, 2f, 11f, 7f)
        lineTo(11f, 12f)
        curveTo(11f, 16f, 5f, 16f, 5f, 12f)
        close()
        moveTo(2f, 11f)
        curveTo(2f, 21f, 14f, 21f, 14f, 11f)
        line(8f, 19f, 8f, 22f)
        line(16f, 5f, 22f, 5f)
        line(16f, 9f, 20f, 9f)
        line(17f, 13f, 22f, 13f)
      },
    "text-to-speech" to
      glyph("speech-output") {
        moveTo(3f, 3f)
        lineTo(21f, 3f)
        lineTo(21f, 17f)
        lineTo(10f, 17f)
        lineTo(5f, 21f)
        lineTo(5f, 17f)
        lineTo(3f, 17f)
        close()
        line(7f, 8f, 7f, 12f)
        line(10f, 6f, 10f, 14f)
        line(14f, 8f, 14f, 12f)
        line(17f, 7f, 17f, 13f)
      },
    "audio-codec" to
      glyph("codec") {
        box(8f, 7f, 8f, 10f)
        line(2f, 10f, 6f, 10f)
        line(2f, 14f, 6f, 14f)
        line(18f, 10f, 22f, 10f)
        line(18f, 14f, 22f, 14f)
        moveTo(9f, 4f)
        lineTo(12f, 1f)
        lineTo(15f, 4f)
        moveTo(9f, 20f)
        lineTo(12f, 23f)
        lineTo(15f, 20f)
      },
    "audio-classification" to
      glyph("sound-label") {
        line(3f, 8f, 3f, 16f)
        line(7f, 4f, 7f, 20f)
        line(11f, 7f, 11f, 17f)
        line(15f, 10f, 15f, 14f)
        moveTo(17f, 18f)
        lineTo(19f, 20f)
        lineTo(23f, 15f)
        line(19f, 5f, 22f, 5f)
        line(19f, 9f, 22f, 9f)
      },
    "pitch-detection" to
      glyph("pitch") {
        wave(12f)
        line(3f, 20f, 21f, 20f)
        line(12f, 3f, 12f, 7f)
        moveTo(10f, 5f)
        lineTo(12f, 3f)
        lineTo(14f, 5f)
      },
    "audio-source-separation" to
      glyph("stems") {
        line(2f, 12f, 7f, 12f)
        moveTo(7f, 12f)
        lineTo(12f, 5f)
        lineTo(21f, 5f)
        line(7f, 12f, 21f, 12f)
        moveTo(7f, 12f)
        lineTo(12f, 19f)
        lineTo(21f, 19f)
        line(18f, 3f, 18f, 7f)
        line(18f, 10f, 18f, 14f)
        line(18f, 17f, 18f, 21f)
      },
    "speech-enhancement" to
      glyph("enhance") {
        line(3f, 9f, 3f, 15f)
        line(6f, 5f, 6f, 19f)
        line(10f, 8f, 10f, 16f)
        line(14f, 10f, 14f, 14f)
        moveTo(19f, 2f)
        lineTo(20f, 5f)
        lineTo(23f, 6f)
        lineTo(20f, 7f)
        lineTo(19f, 10f)
        lineTo(18f, 7f)
        lineTo(15f, 6f)
        lineTo(18f, 5f)
        close()
        line(18f, 18f, 22f, 18f)
        line(20f, 16f, 20f, 20f)
      },
    "music-transcription" to
      glyph("notes") {
        circle(6f, 18f, 3f)
        circle(18f, 15f, 3f)
        moveTo(9f, 18f)
        lineTo(9f, 5f)
        lineTo(21f, 2f)
        lineTo(21f, 15f)
        line(9f, 9f, 21f, 6f)
      },
    "image-matching" to
      glyph("matching") {
        box(2f, 4f, 7f, 15f)
        box(15f, 4f, 7f, 15f)
        line(6f, 8f, 18f, 14f)
        line(6f, 15f, 18f, 8f)
        circle(6f, 8f, 1f)
        circle(18f, 14f, 1f)
      },
    "image-tagging" to
      glyph("tags") {
        moveTo(3f, 3f)
        lineTo(12f, 3f)
        lineTo(22f, 13f)
        lineTo(13f, 22f)
        lineTo(3f, 12f)
        close()
        circle(8f, 8f, 1.5f)
        line(12f, 10f, 17f, 15f)
      },
    "image-quality" to
      glyph("quality") {
        moveTo(12f, 2f)
        lineTo(15f, 8f)
        lineTo(22f, 9f)
        lineTo(17f, 14f)
        lineTo(18f, 21f)
        lineTo(12f, 18f)
        lineTo(6f, 21f)
        lineTo(7f, 14f)
        lineTo(2f, 9f)
        lineTo(9f, 8f)
        close()
      },
    "image-classification" to
      glyph("classes") {
        circle(6f, 6f, 3f)
        box(15f, 3f, 6f, 6f)
        moveTo(3f, 21f)
        lineTo(6f, 14f)
        lineTo(9f, 21f)
        close()
        moveTo(18f, 14f)
        lineTo(22f, 18f)
        lineTo(18f, 22f)
        lineTo(14f, 18f)
        close()
      },
    "fine-grained-classification" to
      glyph("species") {
        moveTo(4f, 20f)
        curveTo(5f, 7f, 12f, 3f, 21f, 3f)
        curveTo(21f, 14f, 17f, 21f, 4f, 20f)
        close()
        line(4f, 20f, 17f, 7f)
        line(10f, 14f, 10f, 8f)
        line(10f, 14f, 17f, 14f)
      },
    "ocr" to
      glyph("text-box") {
        moveTo(3f, 8f)
        lineTo(3f, 3f)
        lineTo(8f, 3f)
        moveTo(16f, 3f)
        lineTo(21f, 3f)
        lineTo(21f, 8f)
        moveTo(3f, 16f)
        lineTo(3f, 21f)
        lineTo(8f, 21f)
        moveTo(16f, 21f)
        lineTo(21f, 21f)
        lineTo(21f, 16f)
        line(7f, 8f, 17f, 8f)
        line(12f, 8f, 12f, 17f)
        line(9f, 17f, 15f, 17f)
      },
    "monocular-geometry-estimation" to
      glyph("depth") {
        moveTo(2f, 17f)
        lineTo(12f, 22f)
        lineTo(22f, 17f)
        moveTo(2f, 12f)
        lineTo(12f, 17f)
        lineTo(22f, 12f)
        moveTo(2f, 7f)
        lineTo(12f, 2f)
        lineTo(22f, 7f)
        lineTo(12f, 12f)
        close()
      },
  )
}
