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

package com.google.ai.edge.examples.ormbg

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
 * Owns the [BgRemover] and exposes a single [UiState] for the screen. [BgRemover] reuses native
 * input and output buffers across calls, so model creation and every matte run on one confined
 * worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "ormbg.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var bgRemover: BgRemover? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and cut out a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        bgRemover = BgRemover(modelFile.absolutePath)
        removeBackground(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Removes the background from a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        removeBackground(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_remove_failed),
          )
        }
      }
    }
  }

  private fun removeBackground(source: Bitmap) {
    val bgRemover = bgRemover ?: return
    val (alpha, elapsedMs) = bgRemover.matte(source)
    val result = cutout(source, alpha)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = source,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Composite the foreground onto a transparency checkerboard using the alpha matte. */
  private fun cutout(image: Bitmap, alpha: FloatArray): Bitmap {
    val S = BgRemover.SIZE
    val w = image.width
    val h = image.height
    val scaled = Bitmap.createScaledBitmap(image, w, h, true)
    val px = IntArray(w * h)
    scaled.getPixels(px, 0, w, 0, 0, w, h)
    val out = IntArray(w * h)
    for (y in 0 until h) {
      val ay = y * S / h
      for (x in 0 until w) {
        val a = alpha[ay * S + x * S / w]
        val p = px[y * w + x]
        val fr = (p shr 16) and 0xFF
        val fg = (p shr 8) and 0xFF
        val fb = p and 0xFF
        val ck = if (((x / 24) + (y / 24)) % 2 == 0) 255 else 205
        val rr = (fr * a + ck * (1 - a)).toInt()
        val gg = (fg * a + ck * (1 - a)).toInt()
        val bb = (fb * a + ck * (1 - a)).toInt()
        out[y * w + x] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
      }
    }
    return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
  }

  override fun onCleared() {
    super.onCleared()
    bgRemover?.close()
  }
}
