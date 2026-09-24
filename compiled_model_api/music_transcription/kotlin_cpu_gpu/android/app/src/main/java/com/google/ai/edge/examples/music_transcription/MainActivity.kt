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

package com.google.ai.edge.examples.music_transcription

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.music_transcription.view.ApplicationTheme
import com.google.ai.edge.examples.music_transcription.view.PianoRollScreen

/**
 * Basic Pitch on-device music transcription. The model runs on the LiteRT CompiledModel GPU (see
 * [Transcriber]); the UI is a thin Compose host over [MainViewModel]. Record from the mic (after a
 * runtime RECORD_AUDIO grant) or pick an audio clip, and see the notes on a piano roll.
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        val context = LocalContext.current

        val recordPermissionLauncher =
          rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted
            ->
            if (granted) viewModel.recordAndTranscribe()
          }

        val pickAudioLauncher =
          rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) viewModel.transcribeUri(uri)
          }

        PianoRollScreen(
          uiState = uiState,
          onRecord = {
            if (
              ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
            ) {
              viewModel.recordAndTranscribe()
            } else {
              recordPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
          },
          onPick = { pickAudioLauncher.launch("audio/*") },
        )
      }
    }
  }
}
