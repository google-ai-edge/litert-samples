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

package com.google.ai.edge.examples.dinov2

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/**
 * Immutable snapshot of everything the feature-visualization screen renders. [resultImage] is the
 * source image and its DINOv2 feature-PCA overlay drawn side by side.
 */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isProcessing: Boolean = false,
  val resultImage: Bitmap? = null,
  val inferenceTimeMs: Long = 0L,
  val errorMessage: String? = null,
)
