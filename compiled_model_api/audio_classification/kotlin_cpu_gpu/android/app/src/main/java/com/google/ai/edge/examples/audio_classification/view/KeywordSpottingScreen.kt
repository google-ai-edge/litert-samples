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

package com.google.ai.edge.examples.audio_classification.view

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.ai.edge.examples.audio_classification.R
import com.google.ai.edge.examples.audio_classification.UiState

/**
 * Keyword-spotting screen: a status line, the fixed 10-word command set as help text, a Record
 * button, and the recognized keyword. The Record button requests the RECORD_AUDIO permission on
 * demand and only invokes [onRecord] once it is granted.
 */
@Composable
fun KeywordSpottingScreen(uiState: UiState, onRecord: () -> Unit, modifier: Modifier = Modifier) {
  val context = LocalContext.current
  val permissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) onRecord()
    }
  val requestRecord = {
    if (
      ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
        PackageManager.PERMISSION_GRANTED
    ) {
      onRecord()
    } else {
      permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }
  }
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
      Spacer(modifier = Modifier.height(24.dp))
      Text(text = stringResource(R.string.howto), fontSize = 15.sp)
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text = stringResource(R.string.keywords),
        fontSize = 20.sp,
        color = keywordBlue,
        textAlign = TextAlign.Center,
        modifier = Modifier.fillMaxWidth(),
      )
      Spacer(modifier = Modifier.height(8.dp))
      Text(text = stringResource(R.string.command_set_note), fontSize = 12.sp, color = Color.Gray)
      Spacer(modifier = Modifier.weight(1f))
      Text(
        text = uiState.resultText.ifEmpty { stringResource(R.string.result_placeholder) },
        fontSize = 34.sp,
        textAlign = TextAlign.Center,
        color = resultColor(uiState),
        modifier = Modifier.fillMaxWidth(),
      )
      Spacer(modifier = Modifier.weight(1f))
      Button(
        onClick = requestRecord,
        enabled = uiState.isModelReady && !uiState.isRecording,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(text = stringResource(R.string.action_record))
      }
    }
  }
}

/**
 * Picks the color of the large result line: red while listening, green for a recognized keyword,
 * gray for "_unknown_"/"_silence_" or a failure, and the default for the initial placeholder.
 */
private fun resultColor(uiState: UiState): Color =
  when {
    uiState.isRecording -> recordingRed
    uiState.resultText.isEmpty() -> Color.Unspecified
    uiState.isRecognized -> recognizedGreen
    else -> Color.Gray
  }
