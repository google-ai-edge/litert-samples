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

package com.google.ai.edge.examples.inpainting

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** One finger-painted stroke, in the model's SIZE x SIZE image coordinate space. */
@Immutable data class Stroke(val points: List<StrokePoint>)

/** A single sampled point of a [Stroke]. */
@Immutable data class StrokePoint(val x: Float, val y: Float)

/** Immutable snapshot of everything the inpainting screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isProcessing: Boolean = false,
  /** The current working image: the loaded photo, or the last inpainted result. */
  val image: Bitmap? = null,
  val strokes: List<Stroke> = emptyList(),
  /** Non-zero once an inpaint has run on the current image. */
  val inferenceTimeMs: Long = 0L,
  /** True when the user tapped Erase without painting anything first. */
  val isMissingStrokes: Boolean = false,
  val errorMessage: String? = null,
)
