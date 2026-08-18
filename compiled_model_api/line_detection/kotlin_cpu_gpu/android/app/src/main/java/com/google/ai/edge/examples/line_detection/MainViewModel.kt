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

package com.google.ai.edge.examples.line_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
 * Owns the [MlsdDetector] and exposes a single [UiState] for the screen. [MlsdDetector] reuses
 * native input and output buffers across calls, so model creation and every detection run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    // Red overlay stroke matching the original demo's line paint.
    private const val LINE_COLOR_RED = 255
    private const val LINE_COLOR_GREEN = 40
    private const val LINE_COLOR_BLUE = 40
    private const val LINE_STROKE_WIDTH = 3f

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var detector: MlsdDetector? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and detect on a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MlsdDetector.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        detector = MlsdDetector(context)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET), warm = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Detects line segments on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        detect(context.loadOrientedBitmap(uri), warm = false)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_detection_failed),
          )
        }
      }
    }
  }

  /**
   * Center-crops [source] to the model's square input, runs M-LSD, and renders the detected
   * segments over the resized image. The line coordinates come back in the resized image's pixel
   * space, so the overlay is drawn directly onto that square bitmap.
   *
   * When [warm] is true, an untimed detection runs first so the reported time excludes first-run
   * GPU and JIT warm-up overhead. The bundled demo image warms up; gallery picks do not.
   */
  private fun detect(source: Bitmap, warm: Boolean) {
    val detector = detector ?: return
    val square = source.squareResize(MlsdDetector.SIZE)
    val rgb = square.toRgbFloatArray()
    if (warm) {
      detector.detect(rgb, square.width.toFloat(), square.height.toFloat())
    }
    val startNanos = System.nanoTime()
    val lines = detector.detect(rgb, square.width.toFloat(), square.height.toFloat())
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = drawLines(square, lines),
        inferenceTimeMs = elapsedMs,
        segmentCount = lines.size,
      )
    }
  }

  /** Draws the detected [lines] onto a copy of [image] with the demo's red overlay paint. */
  private fun drawLines(image: Bitmap, lines: List<MlsdDetector.Line>): Bitmap {
    val annotated = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(annotated)
    val paint =
      Paint().apply {
        color = Color.rgb(LINE_COLOR_RED, LINE_COLOR_GREEN, LINE_COLOR_BLUE)
        strokeWidth = LINE_STROKE_WIDTH
        isAntiAlias = true
      }
    for (line in lines) {
      canvas.drawLine(line.x1, line.y1, line.x2, line.y2, paint)
    }
    return annotated
  }

  override fun onCleared() {
    super.onCleared()
    detector?.close()
  }
}
