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

package com.google.ai.edge.examples.audio_source_separation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.audio_source_separation.view.ApplicationTheme
import com.google.ai.edge.examples.audio_source_separation.view.AudioSeparationScreen

/**
 * TIGER-DnR cinematic sound separation on the LiteRT CompiledModel GPU (see [TigerSeparator]). The
 * UI is a thin Compose host over [MainViewModel]; mic capture, separation and stem playback all
 * live in the ViewModel, and the mic-permission request lives in [AudioSeparationScreen].
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        AudioSeparationScreen(
          uiState = uiState,
          onPickClip = { viewModel.separateFromUri(it) },
          onRecord = { viewModel.record() },
          onPlay = { viewModel.play(it) },
        )
      }
    }
  }
}
