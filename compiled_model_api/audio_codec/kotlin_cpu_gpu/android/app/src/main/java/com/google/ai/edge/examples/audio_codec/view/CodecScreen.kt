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

package com.google.ai.edge.examples.audio_codec.view

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.audio_codec.R
import com.google.ai.edge.examples.audio_codec.UiState

/**
 * Codec screen: the round-trip status line plus two Play buttons that A/B the bundled original clip
 * against the codec reconstruction.
 *
 * @param uiState the current snapshot to render.
 * @param onPlayOriginal invoked when the user taps "Play original".
 * @param onPlayReconstructed invoked when the user taps "Play reconstructed".
 * @param modifier the modifier applied to the root [Scaffold].
 */
@Composable
fun CodecScreen(
  uiState: UiState,
  onPlayOriginal: () -> Unit,
  onPlayReconstructed: () -> Unit,
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
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
      Spacer(modifier = Modifier.height(16.dp))
      Button(onClick = onPlayOriginal, enabled = uiState.isModelReady) {
        Text(text = stringResource(R.string.action_play_original))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Button(onClick = onPlayReconstructed, enabled = uiState.isModelReady) {
        Text(text = stringResource(R.string.action_play_reconstructed))
      }
    }
  }
}
