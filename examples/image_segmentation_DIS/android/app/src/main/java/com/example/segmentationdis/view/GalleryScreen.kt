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

package com.example.segmentationdis.view

import android.graphics.Bitmap
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.core.graphics.drawable.toBitmap
import coil.compose.AsyncImage
import com.example.segmentationdis.UiState

@Composable
fun GalleryScreen(
    uiState: UiState, modifier: Modifier = Modifier, onImageAnalyzed: (Bitmap) -> Unit
) {
    val uri = uiState.galleryUri

    Box(modifier = modifier) {
        AsyncImage(
            modifier = Modifier.fillMaxSize(),
            model = uri,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            onSuccess = {
                val bimap = it.result.drawable.toBitmap()
                onImageAnalyzed(bimap)
            },
        )
        if (uiState.galleryOverlayInfo != null) {
            SegmentationOverlay(
                modifier = Modifier.fillMaxSize(), overlayInfo = uiState.galleryOverlayInfo
            )
        }
    }
}
