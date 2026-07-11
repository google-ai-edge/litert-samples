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

package com.google.ai.edge.examples.image_restoration

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [NafnetRestorer] and exposes a single [UiState] for the screen. [NafnetRestorer] reuses
 * native input and output buffers across calls, so model creation and every restoration run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var restorer: NafnetRestorer? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and restore a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, NafnetRestorer.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        restorer = NafnetRestorer(context)
        restore(context.decodeAssetBitmap(DEMO_IMAGE_ASSET), warmup = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Restores a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        restore(context.loadOrientedBitmap(uri), warmup = false)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_restore_failed),
          )
        }
      }
    }
  }

  /**
   * Center-crops and resizes [source] to the model's square input, runs NAFNet, and publishes the
   * blurry input and restored output. A [warmup] pass primes the GPU so the timed run is steady.
   */
  private fun restore(source: Bitmap, warmup: Boolean) {
    val restorer = restorer ?: return
    val input = squareResize(source)
    val rgb = bitmapToRgb(input)
    if (warmup) {
      restorer.restore(rgb)
    }
    val startNanos = System.nanoTime()
    val out = restorer.restore(rgb)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    val result = rgbToBitmap(out, NafnetRestorer.SIZE)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = input,
        resultImage = result,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Center-crops [src] to a square and scales it to the model's [NafnetRestorer.SIZE]. */
  private fun squareResize(src: Bitmap): Bitmap {
    val s = min(src.width, src.height)
    val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    return Bitmap.createScaledBitmap(crop, NafnetRestorer.SIZE, NafnetRestorer.SIZE, true)
  }

  /** Flattens [bm] to a row-major RGB float array in [0, 255]. */
  private fun bitmapToRgb(bm: Bitmap): FloatArray {
    val n = bm.width * bm.height
    val px = IntArray(n)
    bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
    val out = FloatArray(n * 3)
    for (i in 0 until n) {
      val p = px[i]
      out[i * 3] = ((p shr 16) and 0xFF).toFloat()
      out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
      out[i * 3 + 2] = (p and 0xFF).toFloat()
    }
    return out
  }

  /** Packs a row-major RGB float array in [0, 255] back into a [size] x [size] bitmap. */
  private fun rgbToBitmap(rgb: FloatArray, size: Int): Bitmap {
    val px = IntArray(size * size)
    for (i in 0 until size * size) {
      px[i] =
        Color.rgb(
          rgb[i * 3].toInt().coerceIn(0, 255),
          rgb[i * 3 + 1].toInt().coerceIn(0, 255),
          rgb[i * 3 + 2].toInt().coerceIn(0, 255),
        )
    }
    return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
  }

  override fun onCleared() {
    super.onCleared()
    restorer?.close()
  }
}
