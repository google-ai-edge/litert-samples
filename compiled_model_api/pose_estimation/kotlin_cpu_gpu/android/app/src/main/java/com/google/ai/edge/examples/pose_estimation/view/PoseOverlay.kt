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

package com.google.ai.edge.examples.pose_estimation.view

import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import com.google.ai.edge.examples.pose_estimation.Keypoint

// Skeleton edges over the 18 lightweight-OpenPose keypoints
// (nose, neck, r/l shoulder-elbow-wrist, r/l hip-knee-ankle, r/l eye, r/l ear).
private val EDGES =
  listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 1 to 5, 5 to 6, 6 to 7,
    1 to 8, 8 to 9, 9 to 10, 1 to 11, 11 to 12, 12 to 13,
    0 to 14, 14 to 16, 0 to 15, 15 to 17,
  )

private const val SCORE_THRESHOLD = 0.1f
private val LINE_COLOR = Color(0xFF00C99E)
private val POINT_COLOR = Color(0xFFFFD54F)

/**
 * Draws the pose skeleton over the input. Keypoints are normalized to the source frame, so they
 * are placed inside the aspect-fit rectangle of [sourceWidth] x [sourceHeight] within the canvas —
 * matching a `ContentScale.Fit` preview/image. Mirrors X for the front camera.
 */
@Composable
fun PoseOverlay(
  keypoints: List<Keypoint>,
  sourceWidth: Int,
  sourceHeight: Int,
  lensFacing: Int,
  modifier: Modifier = Modifier,
) {
  Canvas(modifier = modifier) {
    if (keypoints.isEmpty() || sourceWidth <= 0 || sourceHeight <= 0) return@Canvas

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
    fun point(k: Keypoint): Offset {
      val nx = if (front) 1f - k.x else k.x
      return Offset(offX + nx * rectW, offY + k.y * rectH)
    }

    for ((a, b) in EDGES) {
      val ka = keypoints[a]
      val kb = keypoints[b]
      if (ka.score > SCORE_THRESHOLD && kb.score > SCORE_THRESHOLD) {
        drawLine(LINE_COLOR, point(ka), point(kb), strokeWidth = 8f, cap = StrokeCap.Round)
      }
    }
    for (kp in keypoints) {
      if (kp.score > SCORE_THRESHOLD) {
        drawCircle(POINT_COLOR, radius = 9f, center = point(kp))
      }
    }
  }
}
