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

package com.google.ai.edge.examples.movinet.view

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.movinet.R
import com.google.ai.edge.examples.movinet.UiState

/**
 * Action-recognition screen: a status line, a Run button, and the streaming per-frame prediction log
 * that fills in frame by frame from [UiState.outputText].
 */
@Composable
fun ActionScreen(uiState: UiState, onRun: () -> Unit, modifier: Modifier = Modifier) {
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
      Spacer(modifier = Modifier.height(8.dp))
      Button(onClick = onRun, enabled = uiState.isModelReady && !uiState.isRunning) {
        Text(text = stringResource(R.string.action_run))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text = uiState.outputText,
        fontSize = 15.sp,
        fontFamily = FontFamily.Monospace,
        modifier = Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState()),
      )
    }
  }
}
