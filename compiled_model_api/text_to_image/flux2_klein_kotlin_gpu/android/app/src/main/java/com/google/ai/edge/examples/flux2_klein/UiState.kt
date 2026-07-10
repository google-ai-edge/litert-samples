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

package com.google.ai.edge.examples.flux2_klein

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** Immutable snapshot of everything the screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isEditingAvailable: Boolean = false,
  val isGenerating: Boolean = false,
  val statusMessage: String = "",
  val errorMessage: String? = null,
  val sourceImage: Bitmap? = null,
  val image: Bitmap? = null,
)
