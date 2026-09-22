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

import kotlin.math.roundToInt

/** Presentation geometry only: model inputs, result pixels, and numerical output are unchanged. */
object ImageDisplayGeometry {
  private val fullSceneOutputs =
    setOf(
      "background-removal",
      "super-resolution",
      "super-resolution-real-esrgan",
      "monocular-geometry-estimation",
      "semantic-segmentation",
      "instance-segmentation",
      "video-action-recognition",
      "lane-detection",
      "head-pose-estimation",
      "face-liveness-anti-spoofing",
      "image-dehazing",
      "portrait-matting",
      "crowd-counting",
      "dense-feature-visualization",
    )

  fun resultAspectRatio(
    taskId: String,
    inputWidth: Int?,
    inputHeight: Int?,
    outputWidth: Int,
    outputHeight: Int,
  ): Float {
    require(outputWidth > 0 && outputHeight > 0)
    return if (
      taskId in fullSceneOutputs &&
        inputWidth != null &&
        inputHeight != null &&
        inputWidth > 0 &&
        inputHeight > 0
    )
      inputWidth.toFloat() / inputHeight
    else outputWidth.toFloat() / outputHeight
  }

  /** XFeat displays two scenes side by side; preserve each photo's ratio independently. */
  fun matchingPanelWidth(inputWidth: Int, inputHeight: Int, displayHeight: Int): Int {
    require(inputWidth > 0 && inputHeight > 0 && displayHeight > 0)
    return (displayHeight.toDouble() * inputWidth / inputHeight).roundToInt().coerceAtLeast(1)
  }
}
