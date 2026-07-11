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

package com.google.ai.edge.examples.inpainting.view

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke as StrokeStyle
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.inpainting.MiganInpainter
import com.google.ai.edge.examples.inpainting.R
import com.google.ai.edge.examples.inpainting.Stroke
import com.google.ai.edge.examples.inpainting.UiState

/** Translucent red the painted mask is previewed with, matching the pre-Compose demo. */
private val MASK_OVERLAY_COLOR = Color(red = 255, green = 40, blue = 40, alpha = 140)

/** Brush radius in image space; the mask MainViewModel rasterizes uses the same fraction. */
private const val BRUSH_RADIUS = MiganInpainter.SIZE * 0.06f

/** Top-level inpainting screen: paint over an object, tap Erase, and it is removed on the GPU. */
@Composable
fun InpaintingScreen(
  uiState: UiState,
  onPickImage: () -> Unit,
  onErase: () -> Unit,
  onReset: () -> Unit,
  onStrokeStart: (Float, Float) -> Unit,
  onStrokePoint: (Float, Float) -> Unit,
  modifier: Modifier = Modifier,
) {
  Scaffold(
    modifier = modifier.statusBarsPadding(),
    topBar = {
      TopAppBar(
        backgroundColor = MaterialTheme.colors.secondary,
        title = { Text(text = stringResource(R.string.app_name), color = Color.White) },
      )
    },
  ) { padding ->
    Column(modifier = Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
      StatusHeader(uiState)
      Spacer(modifier = Modifier.height(12.dp))
      ActionRow(uiState, onPickImage, onErase, onReset)
      Spacer(modifier = Modifier.height(12.dp))
      uiState.image?.let { image ->
        Box(modifier = Modifier.fillMaxWidth().aspectRatio(1f)) {
          Image(
            bitmap = image.asImageBitmap(),
            contentDescription = null,
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize(),
          )
          MaskOverlay(uiState.strokes, onStrokeStart, onStrokePoint)
        }
      }
    }
  }
}

@Composable
private fun ActionRow(
  uiState: UiState,
  onPickImage: () -> Unit,
  onErase: () -> Unit,
  onReset: () -> Unit,
) {
  val enabled = uiState.isModelReady && !uiState.isProcessing
  Row(modifier = Modifier.fillMaxWidth()) {
    Button(onClick = onPickImage, enabled = enabled) {
      Text(text = stringResource(R.string.action_pick_image))
    }
    Spacer(modifier = Modifier.width(8.dp))
    Button(onClick = onErase, enabled = enabled) {
      Text(text = stringResource(R.string.action_erase))
    }
    Spacer(modifier = Modifier.width(8.dp))
    Button(onClick = onReset, enabled = enabled) {
      Text(text = stringResource(R.string.action_reset))
    }
  }
}

/**
 * Draws the painted strokes over the image and forwards drag gestures as image-space coordinates.
 * The image is square and fills this box, so the layout-to-image scale is a single ratio.
 */
@Composable
private fun MaskOverlay(
  strokes: List<Stroke>,
  onStrokeStart: (Float, Float) -> Unit,
  onStrokePoint: (Float, Float) -> Unit,
) {
  Canvas(
    modifier =
      Modifier.fillMaxSize().pointerInput(Unit) {
        val layoutToImage = MiganInpainter.SIZE.toFloat() / size.width
        detectDragGestures(
          onDragStart = { offset ->
            onStrokeStart(offset.x * layoutToImage, offset.y * layoutToImage)
          }
        ) { change, _ ->
          onStrokePoint(change.position.x * layoutToImage, change.position.y * layoutToImage)
        }
      }
  ) {
    val imageToLayout = size.width / MiganInpainter.SIZE
    for (stroke in strokes) {
      val path = Path()
      val first = stroke.points.first()
      path.moveTo(first.x * imageToLayout, first.y * imageToLayout)
      for (point in stroke.points.drop(1)) {
        path.lineTo(point.x * imageToLayout, point.y * imageToLayout)
      }
      drawPath(
        path = path,
        color = MASK_OVERLAY_COLOR,
        style =
          StrokeStyle(
            width = BRUSH_RADIUS * 2 * imageToLayout,
            cap = StrokeCap.Round,
            join = StrokeJoin.Round,
          ),
      )
    }
  }
}

@Composable
private fun StatusHeader(uiState: UiState) {
  val statusText =
    when {
      uiState.errorMessage != null -> uiState.errorMessage
      !uiState.isModelReady -> stringResource(R.string.status_loading)
      uiState.isProcessing -> stringResource(R.string.status_erasing)
      uiState.isMissingStrokes -> stringResource(R.string.status_paint_first)
      uiState.inferenceTimeMs > 0L ->
        stringResource(R.string.status_ready, uiState.inferenceTimeMs)
      else -> stringResource(R.string.status_paint_prompt)
    }
  Text(text = statusText, fontSize = 15.sp, fontWeight = FontWeight.Medium)
}
