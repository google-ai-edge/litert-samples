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

package com.google.ai.edge.examples.speaker_diarization.view

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.ai.edge.examples.speaker_diarization.R
import com.google.ai.edge.examples.speaker_diarization.UiState

/** Diarization screen: record/pick controls, the per-speaker timeline, and per-speaker playback. */
@Composable
fun DiarizationScreen(
  uiState: UiState,
  onToggleRecording: () -> Unit,
  onPickClip: (Uri) -> Unit,
  onPlaySpeaker: (Int) -> Unit,
  modifier: Modifier = Modifier,
) {
  val context = LocalContext.current
  val pickLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
      if (uri != null) onPickClip(uri)
    }
  val recordPermissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) onToggleRecording()
    }
  val onRecord = {
    if (
      uiState.isRecording ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
          PackageManager.PERMISSION_GRANTED
    ) {
      onToggleRecording()
    } else {
      recordPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
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
    Column(
      modifier =
        Modifier.fillMaxSize().padding(padding).padding(16.dp).verticalScroll(rememberScrollState())
    ) {
      val busy = uiState.isRecording || uiState.isAnalyzing
      Text(
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
      Spacer(modifier = Modifier.height(12.dp))
      Button(onClick = onRecord, enabled = uiState.isModelReady && !uiState.isAnalyzing) {
        Text(
          text =
            stringResource(
              if (uiState.isRecording) R.string.action_stop else R.string.action_record
            )
        )
      }
      Spacer(modifier = Modifier.height(8.dp))
      Button(
        onClick = { pickLauncher.launch(arrayOf("audio/*", "video/*")) },
        enabled = uiState.isModelReady && !busy,
      ) {
        Text(text = stringResource(R.string.action_pick))
      }
      uiState.timeline?.let { timeline ->
        Spacer(modifier = Modifier.height(16.dp))
        Image(
          bitmap = timeline.asImageBitmap(),
          contentDescription = null,
          contentScale = ContentScale.Fit,
          modifier =
            Modifier.fillMaxWidth().aspectRatio(timeline.width.toFloat() / timeline.height),
        )
      }
      for (row in uiState.speakers) {
        Spacer(modifier = Modifier.height(8.dp))
        Button(onClick = { onPlaySpeaker(row.speaker) }, enabled = !busy) {
          Text(
            text = stringResource(R.string.action_play_speaker, row.speaker + 1, row.seconds),
            color = Color(row.color),
            fontWeight = FontWeight.Medium,
          )
        }
      }
    }
  }
}
