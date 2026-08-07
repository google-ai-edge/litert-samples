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

package com.google.ai.edge.examples.text_to_speech_dia2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.text_to_speech_dia2.view.ApplicationTheme
import com.google.ai.edge.examples.text_to_speech_dia2.view.TtsScreen

/**
 * On-device Dia2-1B two-speaker dialogue TTS on the LiteRT CompiledModel API. Every graph runs on
 * the CPU as fp32 — see [Dia2Synthesizer]. The UI is a thin Compose host over [MainViewModel].
 * The
 * synthesized waveform is played back through an AudioTrack in the ViewModel.
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        TtsScreen(uiState = uiState, onSynthesize = { viewModel.synthesize(it) })
      }
    }
  }
}
