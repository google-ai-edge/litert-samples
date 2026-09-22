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

package com.google.ai.edge.examples.model_zoo.view

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.google.ai.edge.examples.model_zoo.MainViewModel
import com.google.ai.edge.examples.model_zoo.R
import com.google.ai.edge.examples.model_zoo.UiState
import com.google.ai.edge.examples.model_zoo.data.ModelEntry
import com.google.ai.edge.examples.model_zoo.image.ImageDisplayGeometry

@Composable
internal fun PhotoButtons(choosePhoto: () -> Unit, takePhoto: () -> Unit, enabled: Boolean) {
  Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    OutlinedButton(onClick = choosePhoto, enabled = enabled, modifier = Modifier.weight(1f)) {
      Text(stringResource(R.string.pick_image))
    }
    OutlinedButton(onClick = takePhoto, enabled = enabled, modifier = Modifier.weight(1f)) {
      Text(stringResource(R.string.take_photo))
    }
  }
}

@Composable
internal fun SingleImagePanel(
  entry: ModelEntry,
  state: UiState,
  vm: MainViewModel,
  choosePhoto: () -> Unit,
  takePhoto: () -> Unit,
  chooseSecondPhoto: () -> Unit = {},
) {
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    PhotoButtons(choosePhoto, takePhoto, !state.busy)
    if (entry.taskId == "image-matching") {
      OutlinedButton(
        onClick = chooseSecondPhoto,
        enabled = !state.busy,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text("Choose second photo")
      }
      state.secondaryImage?.let { ImagePreview(it, "Second image", compact = true) }
    }
    val input = state.inputImage
    if (input != null) {
      ImagePreview(input, stringResource(R.string.input_image), compact = true)
      Button(
        onClick = { vm.runSingleImage() },
        modifier = Modifier.fillMaxWidth(),
        enabled =
          !state.busy && (entry.taskId != "image-matching" || state.secondaryImage != null),
      ) {
        Text(stringResource(R.string.run_image))
      }
    } else
      Text(stringResource(R.string.single_image_hint), style = MaterialTheme.typography.bodyMedium)
    if (entry.taskId == "super-resolution-real-esrgan")
      Text(
        stringResource(R.string.super_resolution_input_limit),
        style = MaterialTheme.typography.bodySmall,
      )
  }
}

@Composable
internal fun SingleImageResultPanel(entry: ModelEntry, state: UiState) {
  state.outputImage?.let { output ->
    val transparent = remember(output) { output.containsTransparency() }
    val displayAspectRatio =
      ImageDisplayGeometry.resultAspectRatio(
        entry.taskId,
        state.inputImage?.width,
        state.inputImage?.height,
        output.width,
        output.height,
      )
    if (entry.taskId == "background-removal") {
      var background by remember(output) { mutableStateOf("Transparent") }
      ImagePreview(
        output,
        stringResource(R.string.output_image),
        transparent = background == "Transparent",
        displayAspectRatio = displayAspectRatio,
        background =
          when (background) {
            "White" -> Color.White
            "Green" -> Color(0xFF3EAD69)
            else -> null
          },
      )
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf("Transparent", "White", "Green").forEach { option ->
          FilterChip(
            selected = background == option,
            onClick = { background = option },
            label = { Text(option) },
            modifier = Modifier.heightIn(min = 48.dp),
          )
        }
      }
    } else
      ImagePreview(
        output,
        stringResource(R.string.output_image),
        transparent = transparent,
        displayAspectRatio = displayAspectRatio,
      )
  }
  if (state.imageOutputText.isNotBlank())
    SelectionContainer { Text(state.imageOutputText, style = MaterialTheme.typography.bodyLarge) }
}

/** The frame itself has the fitted bitmap's bounds; backgrounds cannot leak into letterboxing. */
@Composable
internal fun FittedImageFrame(
  bitmap: Bitmap,
  compact: Boolean = false,
  displayAspectRatio: Float = bitmap.width.toFloat() / bitmap.height,
  content: @Composable BoxScope.() -> Unit,
) {
  BoxWithConstraints(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
    val ratio = displayAspectRatio
    val displayHeight = (maxWidth / ratio).coerceAtMost(if (compact) 190.dp else 390.dp)
    val displayWidth = displayHeight * ratio
    Box(
      Modifier.width(displayWidth).height(displayHeight).clip(RoundedCornerShape(12.dp)),
      content = content,
    )
  }
}

@Composable
internal fun ImagePreview(
  bitmap: Bitmap,
  description: String,
  compact: Boolean = false,
  transparent: Boolean = false,
  background: Color? = null,
  displayAspectRatio: Float = bitmap.width.toFloat() / bitmap.height,
) {
  FittedImageFrame(bitmap, compact, displayAspectRatio) {
    if (background != null) Box(Modifier.fillMaxSize().background(background))
    else if (transparent)
      Canvas(Modifier.fillMaxSize()) {
        val tile = 12.dp.toPx()
        for (row in 0 until kotlin.math.ceil(size.height / tile).toInt()) {
          for (column in 0 until kotlin.math.ceil(size.width / tile).toInt()) {
            drawRect(
              if ((row + column) % 2 == 0) Color(0xFFE2E5E8) else Color(0xFFF8F9FA),
              Offset(column * tile, row * tile),
              Size(tile, tile),
            )
          }
        }
      }
    Image(
      bitmap.asImageBitmap(),
      description,
      Modifier.fillMaxSize(),
      // A model may squash the whole photo into a square tensor. Restore only its display
      // geometry here; retained model pixels and inference/pre/post-processing stay unchanged.
      contentScale = ContentScale.FillBounds,
    )
  }
}

private fun Bitmap.containsTransparency(): Boolean {
  if (!hasAlpha()) return false
  val row = IntArray(width)
  for (y in 0 until height) {
    getPixels(row, 0, width, 0, y, width, 1)
    if (row.any { (it ushr 24) < 255 }) return true
  }
  return false
}
