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

package com.google.ai.edge.examples.text_prompted_segmentation.view

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.text_prompted_segmentation.R
import com.google.ai.edge.examples.text_prompted_segmentation.UiState

/** Top-level screen: pick an image, type what to segment, and see the mask blended over it. */
@Composable
fun SegmentationScreen(
  uiState: UiState,
  onPickImage: () -> Unit,
  onPromptChange: (String) -> Unit,
  onSegment: () -> Unit,
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
      OutlinedTextField(
        value = uiState.prompt,
        onValueChange = onPromptChange,
        label = { Text(text = stringResource(R.string.prompt_label)) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
      )
      Spacer(modifier = Modifier.height(12.dp))
      Row {
        Button(onClick = onPickImage, enabled = uiState.isModelReady && !uiState.isProcessing) {
          Text(text = stringResource(R.string.action_pick_image))
        }
        Spacer(modifier = Modifier.width(8.dp))
        Button(onClick = onSegment, enabled = uiState.isModelReady && !uiState.isProcessing) {
          Text(text = stringResource(R.string.action_segment))
        }
      }
      Spacer(modifier = Modifier.height(12.dp))
      val displayed = uiState.resultImage ?: uiState.sourceImage
      displayed?.let { image ->
        Image(
          bitmap = image.asImageBitmap(),
          contentDescription = null,
          contentScale = ContentScale.Fit,
          modifier = Modifier.fillMaxWidth().weight(1f),
        )
      }
    }
  }
}

@Composable
private fun StatusHeader(uiState: UiState) {
  val statusText =
    when {
      uiState.errorMessage != null -> uiState.errorMessage
      !uiState.isModelReady -> stringResource(R.string.status_loading)
      uiState.isProcessing -> stringResource(R.string.status_segmenting, uiState.prompt)
      uiState.isMissingImage -> stringResource(R.string.status_pick_first)
      uiState.resultImage != null ->
        stringResource(R.string.status_ready, uiState.segmentedPrompt, uiState.inferenceTimeMs)
      uiState.sourceImage != null ->
        stringResource(
          R.string.status_image_set,
          uiState.sourceImage.width,
          uiState.sourceImage.height,
        )
      else -> stringResource(R.string.status_pick_prompt)
    }
  Text(text = statusText, fontSize = 15.sp, fontWeight = FontWeight.Medium)
}
