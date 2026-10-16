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

package com.google.ai.edge.examples.dis

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
 * Owns the [CutoutSegmenter] and exposes a single [UiState] for the screen. The DIS model is 176 MB
 * and reuses native buffers, so both model creation and every matte run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "dis.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var segmenter: CutoutSegmenter? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and cut out a bundled image.
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
        segmenter = CutoutSegmenter(modelFile.absolutePath)
        cutout(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs the cutout on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        cutout(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Cutout failed")
        }
      }
    }
  }

  private fun cutout(source: Bitmap) {
    val segmenter = segmenter ?: return
    val (alpha, elapsedMs) = segmenter.matte(source)
    val result = compositeOnCheckerboard(source, alpha, CutoutSegmenter.SIZE)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Composites the object onto a transparency checkerboard using the SIZE x SIZE alpha matte. */
  private fun compositeOnCheckerboard(image: Bitmap, alpha: FloatArray, size: Int): Bitmap {
    val width = image.width
    val height = image.height
    val source = IntArray(width * height)
    image.copy(Bitmap.Config.ARGB_8888, false).getPixels(source, 0, width, 0, 0, width, height)
    val out = IntArray(width * height)
    for (y in 0 until height) {
      val alphaRow = y * size / height
      for (x in 0 until width) {
        val a = alpha[alphaRow * size + x * size / width]
        val pixel = source[y * width + x]
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        val check = if (((x / 24) + (y / 24)) % 2 == 0) 255 else 205
        out[y * width + x] =
          (0xFF shl 24) or
            ((r * a + check * (1 - a)).toInt() shl 16) or
            ((g * a + check * (1 - a)).toInt() shl 8) or
            (b * a + check * (1 - a)).toInt()
      }
    }
    return Bitmap.createBitmap(out, width, height, Bitmap.Config.ARGB_8888)
  }

  override fun onCleared() {
    super.onCleared()
    segmenter?.close()
  }
}
