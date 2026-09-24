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

package com.google.ai.edge.examples.twinlitenet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
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
 * Owns the [TwinLiteSegmenter] and exposes a single [UiState] for the screen. The model reuses
 * native buffers, so both model creation and every segmentation run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "twinlite.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var segmenter: TwinLiteSegmenter? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and segment a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        _uiState.update {
          it.copy(
            errorMessage =
              "Model not found. Push it first with install_to_device.sh:\n" + modelFile.absolutePath
          )
        }
        return@launch
      }
      try {
        segmenter = TwinLiteSegmenter(modelFile.absolutePath)
        segment(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs the segmentation on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        segment(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Segmentation failed")
        }
      }
    }
  }

  private fun segment(source: Bitmap) {
    val segmenter = segmenter ?: return
    val (da, ll, elapsedMs) = segmenter.segment(source)
    val result = render(source, da, ll)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Green = drivable area, red = lane lines, scaled from the W x H masks. */
  private fun render(image: Bitmap, da: ByteArray, ll: ByteArray): Bitmap {
    val W = TwinLiteSegmenter.W
    val H = TwinLiteSegmenter.H
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val px = IntArray(out.width * out.height)
    out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
    for (y in 0 until out.height) {
      val my = y * H / out.height
      for (x in 0 until out.width) {
        val mi = my * W + (x * W / out.width)
        val p = px[y * out.width + x]
        px[y * out.width + x] =
          when {
            ll[mi].toInt() == 1 -> Color.rgb(255, 48, 48)
            da[mi].toInt() == 1 -> {
              val r = (p shr 16) and 0xFF
              val g = (p shr 8) and 0xFF
              val b = p and 0xFF
              Color.rgb(
                (r * 0.45f + 40 * 0.55f).toInt(),
                (g * 0.45f + 220 * 0.55f).toInt(),
                (b * 0.45f + 90 * 0.55f).toInt(),
              )
            }
            else -> p
          }
      }
    }
    return Bitmap.createBitmap(px, out.width, out.height, Bitmap.Config.ARGB_8888)
  }

  override fun onCleared() {
    super.onCleared()
    segmenter?.close()
  }
}
