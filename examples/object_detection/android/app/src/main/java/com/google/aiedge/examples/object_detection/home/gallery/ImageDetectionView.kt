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

package com.google.aiedge.examples.object_detection.home.gallery

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.aiedge.examples.object_detection.UiState
import com.google.aiedge.examples.object_detection.composables.ResultsOverlay
import com.google.aiedge.examples.object_detection.utils.getFittedBoxSize

// ImageDetectionView detects objects in an image and then displays that image with a results
// overlay on top of it

// It takes as an input the object detection options, an image uri, and function to set the
// inference time state
@RequiresApi(Build.VERSION_CODES.P)
@Composable
fun ImageDetectionView(
    uiState: UiState,
    imageUri: Uri,
    modifier: Modifier = Modifier,
    onImageAnalyzed: (Bitmap) -> Unit,
) {
    val context = LocalContext.current

    // We first define some states to hold the results and the image information after being loaded
    val loadedImage by remember {
        val source = ImageDecoder.createSource(context.contentResolver, imageUri)
        mutableStateOf(ImageDecoder.decodeBitmap(source))
    }
    LaunchedEffect(loadedImage) {
        onImageAnalyzed(loadedImage)
    }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize(),
        contentAlignment = Alignment.TopCenter,
    ) {
        // We check if the image is loaded, then we display it
        val imageBitmap = loadedImage.asImageBitmap()

        // When displaying the image, we want to scale it to fit in the available space, filling as
        // much space as it can with being cropped. While this behavior is easily achieved out of
        // the box with composables, we need the results overlay layer to have the exact same size
        // of the rendered image so that the results are drawn correctly on top of it. So we'll have
        // to calculate the size of the image after being scaled to fit in the available space
        // manually. To do that, we use the "getFittedBoxSize" function. Go to its implementation
        // for an explanation of how it works.

        val boxSize = getFittedBoxSize(
            containerSize = Size(
                width = this.maxWidth.value,
                height = this.maxHeight.value,
            ),
            boxSize = Size(
                width = loadedImage.width.toFloat(),
                height = loadedImage.height.toFloat()
            )
        )

        // Now that we have the exact UI size, we display the image and the results
        Box(
            modifier = Modifier
                .width(boxSize.width.dp)
                .height(boxSize.height.dp)
        ) {
            Image(
                bitmap = imageBitmap,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
            )
            uiState.detectionResult?.let {
                ResultsOverlay(
                    result = it
                )
            }
        }
    }

}