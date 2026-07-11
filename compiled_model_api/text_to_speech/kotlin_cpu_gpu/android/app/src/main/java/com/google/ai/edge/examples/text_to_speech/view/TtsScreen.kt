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

package com.google.ai.edge.examples.text_to_speech.view

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.text_to_speech.MainViewModel
import com.google.ai.edge.examples.text_to_speech.R
import com.google.ai.edge.examples.text_to_speech.UiState

/** Text-to-speech screen: a text field, a synthesize button, and the synthesis status. */
@Composable
fun TtsScreen(uiState: UiState, onSynthesize: (String) -> Unit, modifier: Modifier = Modifier) {
  var text by remember { mutableStateOf(MainViewModel.DEFAULT_TEXT) }
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
      OutlinedTextField(
        value = text,
        onValueChange = { text = it },
        label = { Text(text = stringResource(R.string.text_hint)) },
        modifier = Modifier.fillMaxWidth(),
      )
      Spacer(modifier = Modifier.height(8.dp))
      Button(
        onClick = { onSynthesize(text) },
        enabled = uiState.isModelReady && !uiState.isSynthesizing,
      ) {
        Text(text = stringResource(R.string.action_speak))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
    }
  }
}
