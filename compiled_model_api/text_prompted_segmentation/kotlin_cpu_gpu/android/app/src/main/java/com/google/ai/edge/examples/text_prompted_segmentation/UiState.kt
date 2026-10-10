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

package com.google.ai.edge.examples.text_prompted_segmentation

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** Immutable snapshot of everything the text-prompted segmentation screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isProcessing: Boolean = false,
  /** What the user typed; sent to CLIP's text encoder as the segmentation prompt. */
  val prompt: String = MainViewModel.DEFAULT_PROMPT,
  /** The picked image, shown until a mask overlay replaces it. */
  val sourceImage: Bitmap? = null,
  /** The picked image with the predicted mask blended over it. */
  val resultImage: Bitmap? = null,
  /** The prompt the displayed [resultImage] was segmented with. */
  val segmentedPrompt: String = "",
  val inferenceTimeMs: Long = 0L,
  /** True when the user tapped Segment before picking an image. */
  val isMissingImage: Boolean = false,
  val errorMessage: String? = null,
)
