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

package com.google.ai.edge.examples.zero_shot_classification

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.zero_shot_classification.view.ApplicationTheme
import com.google.ai.edge.examples.zero_shot_classification.view.ClassificationScreen

/** Thin Compose host; the ViewModel owns initialization and inference. */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    val launchedAtNs = System.nanoTime()
    super.onCreate(savedInstanceState)
    viewModel.start(accelerator = intent.getStringExtra("accel"), launchedAtNs = launchedAtNs)
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        ClassificationScreen(
          state = state,
          onSubjectChange = viewModel::setSubject,
          onInputChange = viewModel::setInputText,
          onLanguage = viewModel::selectLanguage,
          onPreset = viewModel::selectPreset,
          onAccelerator = viewModel::selectAccelerator,
          onCalibration = viewModel::setCalibrated,
          onRun = viewModel::run,
          onDownload = viewModel::downloadModelFiles,
        )
      }
    }
  }
}
