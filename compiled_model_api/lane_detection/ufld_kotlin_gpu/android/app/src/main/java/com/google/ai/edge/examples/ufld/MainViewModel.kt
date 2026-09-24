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

package com.google.ai.edge.examples.ufld

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
 * Owns the [LaneDetector] and exposes a single [UiState] for the screen. [LaneDetector] reuses
 * native input and output buffers across calls, so model creation and every detection run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "ufld.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var laneDetector: LaneDetector? = null

  /** One color per lane index, matching the four CULane lanes. */
  private val laneColors =
    intArrayOf(
      Color.rgb(255, 60, 60),
      Color.rgb(60, 255, 60),
      Color.rgb(60, 120, 255),
      Color.rgb(255, 220, 40),
    )

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and detect on a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        laneDetector = LaneDetector(modelFile.absolutePath)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Detects lanes in a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        detect(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_detect_failed),
          )
        }
      }
    }
  }

  private fun detect(source: Bitmap) {
    val detector = laneDetector ?: return
    val (points, elapsedMs) = detector.detect(source)
    val annotated = draw(source, points)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = annotated,
        inferenceTimeMs = elapsedMs,
        lanePointCount = points.size,
      )
    }
  }

  /** Draws each detected lane point as a filled circle onto a copy of [image], colored per lane. */
  private fun draw(image: Bitmap, points: List<LanePoint>): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val r = out.width / 90f
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    for (pt in points) {
      paint.color = laneColors[pt.lane % laneColors.size]
      canvas.drawCircle(pt.x * out.width, pt.y * out.height, r, paint)
    }
    return out
  }

  override fun onCleared() {
    super.onCleared()
    laneDetector?.close()
  }
}
