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

package com.google.ai.edge.examples.crowdcount

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [CrowdCounter] and exposes a single [UiState] for the screen. [CrowdCounter] reuses
 * native input and output buffers across calls, so model creation and every count run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "dmcount.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var counter: CrowdCounter? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and count a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        counter = CrowdCounter(modelFile.absolutePath)
        count(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Counts the crowd in a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        count(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_count_failed),
          )
        }
      }
    }
  }

  private fun count(source: Bitmap) {
    val counter = counter ?: return
    val result = counter.count(source)
    val overlay = renderOverlay(source, result.density)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = source,
        resultImage = overlay,
        peopleCount = result.count.roundToInt(),
        inferenceTimeMs = result.inferenceMs,
      )
    }
  }

  /** Composites the per-frame-normalized density map over [input] as a red heatmap. */
  private fun renderOverlay(input: Bitmap, density: FloatArray): Bitmap {
    val side = CrowdCounter.OUT
    var maxV = 1e-5f
    for (v in density) {
      if (v > maxV) {
        maxV = v
      }
    }
    val heatPixels = IntArray(side * side)
    for (i in heatPixels.indices) {
      val v = (density[i] / maxV).coerceIn(0f, 1f)
      val alpha = (v * 220).toInt()
      val green = ((1f - v) * 160).toInt() // faint orange -> strong red
      heatPixels[i] = (alpha shl 24) or (0xFF shl 16) or (green shl 8)
    }
    val heat = Bitmap.createBitmap(heatPixels, side, side, Bitmap.Config.ARGB_8888)
    val out = input.copy(Bitmap.Config.ARGB_8888, true)
    Canvas(out).drawBitmap(
      heat,
      null,
      Rect(0, 0, out.width, out.height),
      Paint(Paint.FILTER_BITMAP_FLAG),
    )
    return out
  }

  override fun onCleared() {
    super.onCleared()
    counter?.close()
  }
}
