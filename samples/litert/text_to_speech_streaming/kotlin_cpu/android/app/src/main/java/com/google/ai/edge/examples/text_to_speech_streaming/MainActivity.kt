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

package com.google.ai.edge.examples.text_to_speech_streaming

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.text_to_speech_streaming.view.ApplicationTheme
import com.google.ai.edge.examples.text_to_speech_streaming.view.TtsScreen

/**
 * On-device KittenTTS nano with sentence-level streaming on the LiteRT CPU interpreter (dynamic
 * sequence length — see [KittenSynthesizer]). The UI is a thin Compose host over [MainViewModel];
 * audio streams through an AudioTrack in the ViewModel while later sentences still synthesize.
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        TtsScreen(
          uiState = uiState,
          onSpeak = { viewModel.speak(it) },
          onStop = { viewModel.stop() },
          onSelectVoice = { viewModel.selectVoice(it) },
        )
      }
    }
  }
}
