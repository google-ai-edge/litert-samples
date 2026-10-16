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

package com.google.ai.edge.examples.gaze_estimation

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
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [GazeEstimator] and exposes a single [UiState] for the screen. [GazeEstimator] reuses
 * native input and output buffers across calls, so model creation and every estimate run on one
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

  private var estimator: GazeEstimator? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model from filesDir (see install_to_device.sh) and warm up on a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, GazeEstimator.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        estimator = GazeEstimator(context)
        estimate(context.decodeAssetBitmap(DEMO_IMAGE_ASSET), warmUp = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Estimates gaze on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        estimate(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_estimate_failed),
          )
        }
      }
    }
  }

  /**
   * Center-crops [source] to a face square, runs L2CS-Net, and publishes the annotated result. When
   * [warmUp] is true a discarded inference primes the GPU so the reported time excludes warm-up.
   */
  private fun estimate(source: Bitmap, warmUp: Boolean = false) {
    val estimator = estimator ?: return
    val face = squareResize(source)
    val rgb = toRgbFloatArray(face)
    if (warmUp) {
      estimator.estimate(rgb)
    }
    val start = System.nanoTime()
    val gaze = estimator.estimate(rgb)
    val elapsedMs = (System.nanoTime() - start) / 1_000_000
    val result = drawGaze(face, gaze)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = result,
        yawDeg = gaze.yawDeg,
        pitchDeg = gaze.pitchDeg,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /**
   * Draws the gaze-direction arrow onto a copy of the [face] crop. The arrow is anchored at the
   * upper-center (~face) of the image and points along the yaw/pitch projected onto the image
   * plane. Ported from the original custom view's `onDraw`, evaluated in the bitmap's own pixel
   * space (a fit-center of a same-aspect bitmap, so the offset is zero and the scale is one).
   */
  private fun drawGaze(face: Bitmap, gaze: GazeEstimator.Gaze): Bitmap {
    val annotated = face.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(annotated)
    val arrowPaint =
      Paint().apply {
        color = Color.rgb(255, 40, 40)
        strokeWidth = 9f
        isAntiAlias = true
      }
    val dotPaint =
      Paint().apply {
        color = Color.rgb(0, 200, 0)
        isAntiAlias = true
      }
    val width = annotated.width.toFloat()
    val height = annotated.height.toFloat()
    val cx = width / 2
    val cy = height * 0.4f
    val len = width * 0.32f
    val yaw = Math.toRadians(gaze.yawDeg.toDouble())
    val pitch = Math.toRadians(gaze.pitchDeg.toDouble())
    val dx = (-len * sin(yaw) * cos(pitch)).toFloat()
    val dy = (-len * sin(pitch)).toFloat()
    canvas.drawLine(cx, cy, cx + dx, cy + dy, arrowPaint)
    canvas.drawCircle(cx, cy, 9f, dotPaint)
    return annotated
  }

  override fun onCleared() {
    super.onCleared()
    estimator?.close()
  }
}
