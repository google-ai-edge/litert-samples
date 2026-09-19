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

package com.google.ai.edge.examples.image_matching.view

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.image_matching.R
import com.google.ai.edge.examples.image_matching.UiState

/**
 * Top-level image-matching screen: a status header, two image pickers, and the side-by-side result
 * image with match lines drawn on top.
 */
@Composable
fun MatchScreen(
  uiState: UiState,
  onPickFirst: () -> Unit,
  onPickSecond: () -> Unit,
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
      val pickersEnabled = uiState.isModelReady && !uiState.isProcessing
      Button(onClick = onPickFirst, enabled = pickersEnabled) {
        Text(text = stringResource(R.string.action_pick_first))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Button(onClick = onPickSecond, enabled = pickersEnabled) {
        Text(text = stringResource(R.string.action_pick_second))
      }
      Spacer(modifier = Modifier.height(12.dp))
      val resultImage = uiState.resultImage
      if (resultImage != null) {
        Image(
          bitmap = resultImage.asImageBitmap(),
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
      uiState.isProcessing -> stringResource(R.string.status_processing)
      uiState.resultImage != null ->
        stringResource(
          R.string.status_result,
          uiState.matchCount,
          uiState.keypointCountA,
          uiState.keypointCountB,
          uiState.inferenceTimeMs,
        )
      uiState.lastPickedSlot > 0 ->
        stringResource(
          R.string.status_image_set,
          uiState.lastPickedSlot,
          uiState.lastPickedWidth,
          uiState.lastPickedHeight,
        )
      else -> stringResource(R.string.status_ready)
    }
  Text(text = statusText, fontSize = 15.sp, fontWeight = FontWeight.Medium)
}
