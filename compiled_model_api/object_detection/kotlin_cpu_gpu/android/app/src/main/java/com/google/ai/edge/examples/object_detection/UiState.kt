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

package com.google.ai.edge.examples.object_detection

import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.compose.runtime.Immutable

@Immutable
class UiState(
  val mediaUri: Uri = Uri.EMPTY,
  // Detected object boxes (normalized to the source frame) + the source aspect ratio, so the
  // boxes can be drawn aligned with the displayed (aspect-fit) input.
  val detections: List<Detection> = emptyList(),
  val sourceWidth: Int = 0,
  val sourceHeight: Int = 0,
  val inferenceTime: Long = 0L,
  val errorMessage: String? = null,
  val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
)

/** A detected object box with corners normalized to `[0,1]` (relative to the source frame). */
@Immutable
data class Detection(
  val left: Float,
  val top: Float,
  val right: Float,
  val bottom: Float,
  val classId: Int,
  val label: String,
  val score: Float,
)
