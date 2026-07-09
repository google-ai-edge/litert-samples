/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.portrait

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [PortraitSketcher] and exposes a single [UiState] for the screen. The U²-Net model is
 * 176 MB and reuses native buffers, so both model creation and every sketch run on one confined
 * worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "portrait.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var sketcher: PortraitSketcher? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and sketch a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        _uiState.update {
          it.copy(
            errorMessage =
              "Model not found. Push it first with install_to_device.sh:\n" +
                modelFile.absolutePath
          )
        }
        return@launch
      }
      try {
        sketcher = PortraitSketcher(modelFile.absolutePath)
        sketch(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs the sketch on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        sketch(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update { it.copy(isProcessing = false, errorMessage = t.message ?: "Sketch failed") }
      }
    }
  }

  /** Runs the pencil-sketch inference; [PortraitSketcher] returns the finished result bitmap. */
  private fun sketch(source: Bitmap) {
    val sketcher = sketcher ?: return
    val (result, elapsedMs) = sketcher.sketch(source)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  override fun onCleared() {
    super.onCleared()
    sketcher?.close()
  }
}
