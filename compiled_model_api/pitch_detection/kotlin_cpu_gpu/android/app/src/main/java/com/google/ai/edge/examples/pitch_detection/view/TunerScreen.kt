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

package com.google.ai.edge.examples.pitch_detection.view

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.ai.edge.examples.pitch_detection.R
import com.google.ai.edge.examples.pitch_detection.UiState

private val inTuneGreen = Color(0xFF2E7D32)
private val offPitchOrange = Color(0xFFE67E22)
private val restingNote = Color(0xFF1A1A1A)

/** Tuner screen: a status line, the detected note, a cents gauge, the Hz reading, and a toggle. */
@Composable
fun TunerScreen(uiState: UiState, onToggleListening: () -> Unit, modifier: Modifier = Modifier) {
  val context = LocalContext.current
  val permissionLauncher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) onToggleListening()
    }
  val onToggle = {
    if (
      uiState.isListening ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
          PackageManager.PERMISSION_GRANTED
    ) {
      onToggleListening()
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
    Column(
      modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
      verticalArrangement = Arrangement.Center,
    ) {
      Text(
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        textAlign = TextAlign.Center,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
      Spacer(modifier = Modifier.height(48.dp))
      Text(
        text = if (uiState.hasPitch) uiState.note else stringResource(R.string.no_pitch),
        fontSize = 96.sp,
        fontWeight = FontWeight.Bold,
        color =
          when {
            !uiState.hasPitch -> Color.LightGray
            uiState.isInTune -> inTuneGreen
            else -> restingNote
          },
      )
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text = uiState.centsGauge,
        fontFamily = FontFamily.Monospace,
        fontSize = 22.sp,
        color = if (uiState.isInTune) inTuneGreen else offPitchOrange,
      )
      Spacer(modifier = Modifier.height(24.dp))
      Text(text = uiState.hzText, fontSize = 18.sp, color = Color.Gray)
      Spacer(modifier = Modifier.height(56.dp))
      Button(onClick = onToggle, enabled = uiState.isModelReady) {
        Text(
          text =
            stringResource(
              if (uiState.isListening) R.string.action_stop else R.string.action_start
            )
        )
      }
    }
  }
}
