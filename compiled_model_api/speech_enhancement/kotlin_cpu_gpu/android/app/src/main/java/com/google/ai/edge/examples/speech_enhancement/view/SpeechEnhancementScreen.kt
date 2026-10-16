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

package com.google.ai.edge.examples.speech_enhancement.view

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.ai.edge.examples.speech_enhancement.R
import com.google.ai.edge.examples.speech_enhancement.UiState

/**
 * Speech-enhancement screen: a status line, a mic Record/Stop toggle, an audio/video picker, and
 * two A/B Play buttons for the noisy and enhanced clips. The Record button requests the
 * RECORD_AUDIO permission on demand and only invokes [onToggleRecord] once it is granted; the
 * picker returns a `content://` [Uri] straight to [onPickAudio].
 */
@Composable
fun SpeechEnhancementScreen(
  uiState: UiState,
  onToggleRecord: () -> Unit,
  onPickAudio: (Uri) -> Unit,
  onPlayNoisy: () -> Unit,
  onPlayEnhanced: () -> Unit,
  modifier: Modifier = Modifier,
) {
  val context = LocalContext.current
  val permissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) onToggleRecord()
    }
  val requestRecord = {
    if (
      ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
        PackageManager.PERMISSION_GRANTED
    ) {
      onToggleRecord()
    } else {
      permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }
  }
  val pickLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
      if (uri != null) onPickAudio(uri)
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
      Spacer(modifier = Modifier.height(16.dp))
      val recordLabel = if (uiState.isRecording) R.string.action_stop else R.string.action_record
      Button(onClick = requestRecord, enabled = uiState.isModelReady && !uiState.isEnhancing) {
        Text(text = stringResource(recordLabel))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Button(
        onClick = { pickLauncher.launch(arrayOf("audio/*", "video/*")) },
        enabled = uiState.isModelReady && !uiState.isRecording && !uiState.isEnhancing,
      ) {
        Text(text = stringResource(R.string.action_pick))
      }
      Spacer(modifier = Modifier.height(24.dp))
      Button(onClick = onPlayNoisy, enabled = uiState.noisy != null) {
        Text(text = stringResource(R.string.action_play_noisy))
      }
      Spacer(modifier = Modifier.height(8.dp))
      Button(onClick = onPlayEnhanced, enabled = uiState.clean != null) {
        Text(text = stringResource(R.string.action_play_enhanced))
      }
    }
  }
}
