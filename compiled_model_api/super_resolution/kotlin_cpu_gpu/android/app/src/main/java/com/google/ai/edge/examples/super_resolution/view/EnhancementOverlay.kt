/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.super_resolution.view

import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Image
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale

/**
 * Draws a result bitmap (here, the Input vs VDSR comparison) preserving its aspect ratio (letterboxed
 * within [modifier]'s bounds). The output bitmap is produced at the input aspect ratio
 * by `SuperResolutionHelper`, so `ContentScale.Fit` keeps it from being stretched.
 */
@Composable
fun EnhancementOverlay(modifier: Modifier = Modifier, overlay: Bitmap, lensFacing: Int) {
  Image(
    bitmap = overlay.asImageBitmap(),
    contentDescription = null,
    contentScale = ContentScale.Fit,
    modifier =
      if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
        // Mirror horizontally to match the front-camera preview.
        modifier.graphicsLayer(scaleX = -1f)
      } else {
        modifier
      },
  )
}
