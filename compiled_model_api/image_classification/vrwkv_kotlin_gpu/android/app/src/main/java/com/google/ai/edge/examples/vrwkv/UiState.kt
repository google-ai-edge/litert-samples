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

package com.google.ai.edge.examples.vrwkv

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** Immutable snapshot of everything the classification screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isProcessing: Boolean = false,
  val sourceImage: Bitmap? = null,
  /** Top-5 ImageNet predictions, most probable first. */
  val predictions: List<VrwkvClassifier.Prediction> = emptyList(),
  val inferenceTimeMs: Long = 0L,
  val errorMessage: String? = null,
)
