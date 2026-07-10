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

package com.google.ai.edge.examples.flux2_klein.view

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.flux2_klein.Flux2KleinGenerator
import com.google.ai.edge.examples.flux2_klein.R
import com.google.ai.edge.examples.flux2_klein.UiState

/**
 * The prompt, the two actions, the status line, and the images.
 *
 * "Generate" runs text-to-image. "Edit an image" opens the photo picker; the picked image is edited
 * with [Flux2KleinGenerator.EDIT_PROMPT]. The edit action only appears when the editing graphs are
 * staged on the device.
 */
@Composable
fun Flux2KleinScreen(
  uiState: UiState,
  onGenerate: () -> Unit,
  onPickImage: () -> Unit,
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
      Text(
        text = "${stringResource(R.string.prompt_hint)}: ${Flux2KleinGenerator.PROMPT}",
        fontSize = 14.sp,
      )
      if (uiState.isEditingAvailable) {
        Text(
          text = "${stringResource(R.string.edit_prompt_hint)}: ${Flux2KleinGenerator.EDIT_PROMPT}",
          fontSize = 14.sp,
        )
      }
      Spacer(modifier = Modifier.height(8.dp))
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = onGenerate, enabled = uiState.isModelReady && !uiState.isGenerating) {
          Text(text = stringResource(R.string.action_generate))
        }
        if (uiState.isEditingAvailable) {
          Button(onClick = onPickImage, enabled = uiState.isModelReady && !uiState.isGenerating) {
            Text(text = stringResource(R.string.action_edit))
          }
        }
      }
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
      uiState.sourceImage?.let { bitmap ->
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = stringResource(R.string.label_source_image), fontSize = 14.sp)
        Image(
          bitmap = bitmap.asImageBitmap(),
          contentDescription = stringResource(R.string.label_source_image),
          modifier = Modifier.fillMaxWidth(),
        )
      }
      uiState.image?.let { bitmap ->
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = stringResource(R.string.label_result_image), fontSize = 14.sp)
        Image(
          bitmap = bitmap.asImageBitmap(),
          contentDescription = stringResource(R.string.label_result_image),
          modifier = Modifier.fillMaxWidth(),
        )
      }
    }
  }
}
