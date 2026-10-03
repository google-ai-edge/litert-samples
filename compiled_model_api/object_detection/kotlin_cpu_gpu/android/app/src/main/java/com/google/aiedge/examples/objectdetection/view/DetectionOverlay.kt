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

package com.google.aiedge.examples.objectdetection.view

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import com.google.aiedge.examples.objectdetection.CocoLabels
import com.google.aiedge.examples.objectdetection.Detection
import kotlin.math.min

/**
 * Draws detection boxes over a fit-center preview. Boxes are in source-image pixels
 * ([sourceWidth] x [sourceHeight]); they are mapped with the same fit-center transform the preview
 * uses (PreviewView FIT_CENTER / AsyncImage Fit), so they line up regardless of view aspect.
 */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    sourceWidth: Int,
    sourceHeight: Int,
    modifier: Modifier = Modifier,
) {
    if (sourceWidth <= 0 || sourceHeight <= 0) return
    val textMeasurer: TextMeasurer = rememberTextMeasurer()
    Canvas(modifier = modifier) {
        val scale = min(size.width / sourceWidth, size.height / sourceHeight)
        val offX = (size.width - sourceWidth * scale) / 2f
        val offY = (size.height - sourceHeight * scale) / 2f
        val strokeWidth = 3.dp.toPx()

        detections.forEach { d ->
            val color = Color(CocoLabels.color(d.classId))
            val left = offX + d.xMin * scale
            val top = offY + d.yMin * scale
            val w = (d.xMax - d.xMin) * scale
            val h = (d.yMax - d.yMin) * scale

            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = Size(w, h),
                style = Stroke(width = strokeWidth),
            )

            val label = "${d.label} ${(d.score * 100).toInt()}%"
            val measured = textMeasurer.measure(label)
            val labelH = measured.size.height.toFloat()
            val labelW = measured.size.width.toFloat()
            val labelTop = (top - labelH).coerceAtLeast(0f)
            drawRect(color = color, topLeft = Offset(left, labelTop), size = Size(labelW, labelH))
            drawText(measured, color = Color.Black, topLeft = Offset(left, labelTop))
        }
    }
}
