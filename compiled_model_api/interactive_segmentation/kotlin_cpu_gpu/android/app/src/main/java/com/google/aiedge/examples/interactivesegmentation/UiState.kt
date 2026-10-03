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

package com.google.aiedge.examples.interactivesegmentation

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
    /** The picked image, already loaded as a bitmap (null until the user picks one). */
    val imageBitmap: Bitmap? = null,
    /** The current best-mask overlay for the last tap (256x256 translucent), or null. */
    val maskBitmap: Bitmap? = null,
    /** Predicted IoU of the displayed mask. */
    val maskIou: Float = 0f,
    /** One-time image encode latency (ms). */
    val encodeTimeMs: Long = 0L,
    /** Per-tap decode latency (ms). */
    val decodeTimeMs: Long = 0L,
    /** True while the encoder is running on a freshly picked image. */
    val isEncoding: Boolean = false,
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val delegate: Sam2SegmentationHelper.AcceleratorEnum = Sam2SegmentationHelper.DEFAULT_DELEGATE,
)
