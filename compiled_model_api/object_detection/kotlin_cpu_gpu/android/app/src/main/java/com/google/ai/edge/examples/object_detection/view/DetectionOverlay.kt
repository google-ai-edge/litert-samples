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

package com.google.ai.edge.examples.object_detection.view

import android.graphics.Paint
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import com.google.ai.edge.examples.object_detection.Detection

private const val STROKE_WIDTH = 6f
private const val TEXT_SIZE = 40f

/** Deterministic per-class color, evenly spread around the hue wheel. */
private fun classColor(classId: Int): Color = Color.hsv((classId * 47 % 360).toFloat(), 0.85f, 1f)

/**
 * Draws the detected object boxes over the input. Boxes are normalized to the source frame, so they
 * are placed inside the aspect-fit rectangle of [sourceWidth] x [sourceHeight] within the canvas —
 * matching a `ContentScale.Fit` preview/image. Mirrors X for the front camera.
 */
@Composable
fun DetectionOverlay(
  detections: List<Detection>,
  sourceWidth: Int,
  sourceHeight: Int,
  lensFacing: Int,
  modifier: Modifier = Modifier,
) {
  val labelPaint = remember {
    Paint().apply {
      color = android.graphics.Color.WHITE
      textSize = TEXT_SIZE
      isAntiAlias = true
      setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK)
    }
  }

  Canvas(modifier = modifier) {
    if (detections.isEmpty() || sourceWidth <= 0 || sourceHeight <= 0) return@Canvas

    val srcAspect = sourceWidth.toFloat() / sourceHeight
    val canvasAspect = size.width / size.height
    val rectW: Float
    val rectH: Float
    val offX: Float
    val offY: Float
    if (srcAspect > canvasAspect) {
      rectW = size.width
      rectH = size.width / srcAspect
      offX = 0f
      offY = (size.height - rectH) / 2f
    } else {
      rectH = size.height
      rectW = size.height * srcAspect
      offX = (size.width - rectW) / 2f
      offY = 0f
    }

    val front = lensFacing == CameraSelector.LENS_FACING_FRONT
    for (d in detections) {
      // Mirror X for the front camera (swap left/right after mirroring).
      val nl = if (front) 1f - d.right else d.left
      val nr = if (front) 1f - d.left else d.right
      val x1 = offX + nl * rectW
      val y1 = offY + d.top * rectH
      val x2 = offX + nr * rectW
      val y2 = offY + d.bottom * rectH

      val color = classColor(d.classId)
      drawRect(
        color = color,
        topLeft = Offset(x1, y1),
        size = Size(x2 - x1, y2 - y1),
        style = Stroke(width = STROKE_WIDTH),
      )

      val text = "${d.label} ${(d.score * 100).toInt()}%"
      drawContext.canvas.nativeCanvas.drawText(text, x1 + 6f, (y1 - 8f).coerceAtLeast(TEXT_SIZE), labelPaint)
    }
  }
}
