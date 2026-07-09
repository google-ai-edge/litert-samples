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

package com.google.ai.edge.examples.flux2_klein

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Loads the [Flux2KleinGenerator] (which reads the staged host inputs) and exposes a single
 * [UiState]. [generate] runs the four-step denoising loop on a confined worker so the GPU graphs
 * are never touched concurrently, and publishes the decoded image.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  private var generator: Flux2KleinGenerator? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState(statusMessage = "Loading FLUX.2-klein…"))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        generator = Flux2KleinGenerator(context)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage = "Ready. Tap Generate to run the chunked transformer on the GPU.",
          )
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load failed", t)
        _uiState.update {
          it.copy(
            errorMessage =
              "${t.message}\n\nStage the graphs first:\n  ./install_to_device.sh <graphs> <bins>"
          )
        }
      }
    }
  }

  /** Runs the denoising loop on the confined worker and publishes the decoded image. */
  fun generate() {
    val model = generator ?: return
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(isGenerating = true, statusMessage = "Generating…", errorMessage = null)
      }
      try {
        val image =
          model.generate { progress -> _uiState.update { it.copy(statusMessage = progress) } }
        _uiState.update { it.copy(isGenerating = false, statusMessage = "Done.", image = image) }
      } catch (t: Throwable) {
        Log.e(TAG, "generate failed", t)
        _uiState.update {
          it.copy(isGenerating = false, errorMessage = t.message ?: "Generation failed")
        }
      }
    }
  }

  override fun onCleared() {
    super.onCleared()
    generator?.close()
    generator = null
  }

  companion object {
    private const val TAG = "Flux2Klein"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }
}
