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

package com.google.ai.edge.examples.movinet

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.movinet.view.ActionScreen
import com.google.ai.edge.examples.movinet.view.ApplicationTheme

/**
 * MoViNet-A0 streaming action recognition on the LiteRT CompiledModel GPU delegate (see [MoViNet]).
 * The UI is a thin Compose host over [MainViewModel], which streams the running per-frame prediction
 * of a bundled reference clip into the screen.
 */
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      ApplicationTheme {
        val uiState by viewModel.uiState.collectAsStateWithLifecycle()
        ActionScreen(uiState = uiState, onRun = { viewModel.run() })
      }
    }
  }
}
