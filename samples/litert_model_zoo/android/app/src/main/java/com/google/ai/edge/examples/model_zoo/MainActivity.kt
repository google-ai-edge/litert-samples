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

package com.google.ai.edge.examples.model_zoo

import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.model_zoo.view.ApplicationTheme
import com.google.ai.edge.examples.model_zoo.view.ModelZooScreen
import java.util.Locale

class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.factory(application) }

  override fun attachBaseContext(newBase: Context) {
    // The app currently ships English only. Keep its resources and plurals in English without
    // changing the device locale, other apps, or the process-wide default used by model code.
    val english =
      Configuration(newBase.resources.configuration).apply {
        setLocale(Locale.ENGLISH)
        setLayoutDirection(Locale.ENGLISH)
      }
    super.attachBaseContext(newBase.createConfigurationContext(english))
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    setContent {
      ApplicationTheme {
        val state = viewModel.state.collectAsStateWithLifecycle()
        ModelZooScreen(state.value, viewModel, this)
      }
    }
  }

  override fun onStop() {
    viewModel.setCamera(false)
    viewModel.stopPlayback()
    viewModel.stopRecording(transcribe = false)
    super.onStop()
  }
}
