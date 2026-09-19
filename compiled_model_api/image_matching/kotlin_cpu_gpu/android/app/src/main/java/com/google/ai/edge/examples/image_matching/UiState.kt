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

package com.google.ai.edge.examples.image_matching

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** Immutable snapshot of everything the image-matching screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isProcessing: Boolean = false,
  /** Slot (1 or 2) of the most recently picked image, or 0 before anything is picked. */
  val lastPickedSlot: Int = 0,
  val lastPickedWidth: Int = 0,
  val lastPickedHeight: Int = 0,
  /** Side-by-side image pair with the match lines drawn on top. */
  val resultImage: Bitmap? = null,
  val matchCount: Int = 0,
  val keypointCountA: Int = 0,
  val keypointCountB: Int = 0,
  val inferenceTimeMs: Long = 0L,
  val errorMessage: String? = null,
)
