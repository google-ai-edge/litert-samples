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

package com.google.ai.edge.examples.dinov2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
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
 * Owns the [Dinov2Features] extractor and exposes a single [UiState] for the screen.
 *
 * [Dinov2Features] loads the fp16 model from the app assets and reuses native input and output
 * buffers across calls, so model creation and every feature run are confined to one single-threaded
 * dispatcher.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "sample.jpg"

    /** Side length (px) of each half of the source-vs-features comparison bitmap. */
    private const val COMPARISON_SIDE = 512

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var extractor: Dinov2Features? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model from assets and visualize a bundled sample image.
    viewModelScope.launch(inferenceDispatcher) {
      try {
        extractor = Dinov2Features(context)
        visualize(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Visualizes the dense features of a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        visualize(context.decodeUriBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_features_failed),
          )
        }
      }
    }
  }

  /** Runs DINOv2, times the GPU inference, and publishes the side-by-side comparison. */
  private fun visualize(source: Bitmap) {
    val extractor = extractor ?: return
    val start = System.nanoTime()
    val featureMap = extractor.featureMap(source)
    val elapsedMs = (System.nanoTime() - start) / 1_000_000
    val comparison = sideBySide(source, featureMap)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = comparison,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Draws the source image and the upscaled feature map side by side. */
  private fun sideBySide(source: Bitmap, featureMap: Bitmap): Bitmap {
    val side = COMPARISON_SIDE
    val out = Bitmap.createBitmap(side * 2, side, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(out)
    canvas.drawBitmap(Bitmap.createScaledBitmap(source, side, side, true), 0f, 0f, null)
    canvas.drawBitmap(
      Bitmap.createScaledBitmap(featureMap, side, side, false),
      side.toFloat(),
      0f,
      null,
    )
    return out
  }

  override fun onCleared() {
    super.onCleared()
    extractor?.close()
  }
}
