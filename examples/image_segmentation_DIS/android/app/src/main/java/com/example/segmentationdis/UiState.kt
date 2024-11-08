/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.example.segmentationdis

import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
    val cameraUri: Uri = Uri.EMPTY,
    val cameraUriTemp: Uri = Uri.EMPTY,
    val galleryUri: Uri = Uri.EMPTY,
    val cameraOverlayInfo: OverlayInfo? = null,
    val galleryOverlayInfo: OverlayInfo? = null,
    val inferenceTime: Long = 0L,
    val errorMessage: String? = null,
    val currentTab: Tab = Tab.Camera
)

@Immutable
enum class Tab {
    Camera, Gallery,
}

@Immutable
data class OverlayInfo(
    val bitmap: Bitmap,
)
