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

package com.google.aiedge.examples.interactivesegmentation.view

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.google.aiedge.examples.interactivesegmentation.R
import com.google.aiedge.examples.interactivesegmentation.Sam2SegmentationHelper
import com.google.aiedge.examples.interactivesegmentation.UiState

private const val MODEL_SIZE = Sam2SegmentationHelper.IMAGE_SIZE.toFloat()

/**
 * Shows the picked image, lets the user tap to segment, and draws the returned mask on top.
 * The image is stretched to fill an aspect-correct box (the same square resize the encoder uses),
 * so a tap fraction maps linearly to 1024x1024 model space and the 256x256 mask overlays exactly.
 */
@Composable
fun SegmentationScreen(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onTap: (Float, Float) -> Unit,
) {
    val image = uiState.imageBitmap
    if (image == null) {
        Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = stringResource(R.string.pick_image), color = Color.Gray)
        }
        return
    }

    var lastTap by remember(image) { mutableStateOf<Offset?>(null) }
    val aspect = image.width.toFloat() / image.height.toFloat()

    Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(aspect)
                .pointerInput(image) {
                    detectTapGestures { offset ->
                        val u = (offset.x / size.width).coerceIn(0f, 1f)
                        val v = (offset.y / size.height).coerceIn(0f, 1f)
                        lastTap = Offset(u, v)
                        onTap(u * MODEL_SIZE, v * MODEL_SIZE)
                    }
                }
        ) {
            Image(
                bitmap = image.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.FillBounds,
            )
            uiState.maskBitmap?.let { mask ->
                Image(
                    bitmap = mask.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds,
                )
            }
            lastTap?.let { f ->
                Canvas(Modifier.fillMaxSize()) {
                    val center = Offset(f.x * size.width, f.y * size.height)
                    drawCircle(Color.White, radius = 11f, center = center)
                    drawCircle(cyan, radius = 7f, center = center)
                }
            }
            if (uiState.isEncoding) {
                Box(
                    Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.3f)),
                    contentAlignment = Alignment.Center,
                ) {
                    CircularProgressIndicator(color = teal)
                }
            }
        }
    }
}
