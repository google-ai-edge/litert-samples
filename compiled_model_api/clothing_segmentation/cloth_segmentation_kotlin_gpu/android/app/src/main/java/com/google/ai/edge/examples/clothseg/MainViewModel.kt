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

package com.google.ai.edge.examples.clothseg

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
 * Owns the [ClothSegmenter] and exposes a single [UiState] for the screen. The U²-Net model reuses
 * native buffers, so both model creation and every segmentation run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "clothseg.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var segmenter: ClothSegmenter? = null

  // 0 bg (keep), 1 upper (cyan), 2 lower (orange), 3 full (magenta).
  private val classColors =
    intArrayOf(0, Color.rgb(0, 200, 255), Color.rgb(255, 150, 0), Color.rgb(230, 0, 200))

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
        segmenter = ClothSegmenter(modelFile.absolutePath)
        runSegmentation(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
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
        runSegmentation(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Segmentation failed")
        }
      }
    }
  }

  private fun runSegmentation(source: Bitmap) {
    val segmenter = segmenter ?: return
    val (classMap, elapsedMs) = segmenter.segment(source)
    val result = overlayClasses(source, classMap, ClothSegmenter.OUT)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Blends a per-class color over the clothing regions using the [size] x [size] class map. */
  private fun overlayClasses(image: Bitmap, cls: ByteArray, size: Int): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val px = IntArray(out.width * out.height)
    out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
    for (y in 0 until out.height) {
      val my = y * size / out.height
      for (x in 0 until out.width) {
        val c = cls[my * size + (x * size / out.width)].toInt()
        if (c != 0) {
          val p = px[y * out.width + x]
          val col = classColors[c]
          val r = (((p shr 16) and 0xFF) * 0.4f + ((col shr 16) and 0xFF) * 0.6f).toInt()
          val g = (((p shr 8) and 0xFF) * 0.4f + ((col shr 8) and 0xFF) * 0.6f).toInt()
          val b = ((p and 0xFF) * 0.4f + (col and 0xFF) * 0.6f).toInt()
          px[y * out.width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
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
