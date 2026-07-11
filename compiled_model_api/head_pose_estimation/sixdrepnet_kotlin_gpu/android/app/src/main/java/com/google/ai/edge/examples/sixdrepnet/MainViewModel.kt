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

package com.google.ai.edge.examples.sixdrepnet

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
 * Owns the [HeadPoseEstimator] and exposes a single [UiState] for the screen. The model is loaded
 * from filesDir and reuses native buffers, so both model creation and every estimate run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "6drepnet.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var estimator: HeadPoseEstimator? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and pose a bundled face.
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
        estimator = HeadPoseEstimator(modelFile.absolutePath)
        estimate(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs head-pose estimation on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        estimate(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Estimation failed")
        }
      }
    }
  }

  private fun estimate(source: Bitmap) {
    val estimator = estimator ?: return
    // centered square crop (assume the face is centered)
    val s = minOf(source.width, source.height)
    val crop = Bitmap.createBitmap(source, (source.width - s) / 2, (source.height - s) / 2, s, s)
    val (pose, elapsedMs) = estimator.estimate(crop)
    val out = drawAxis(crop, pose)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = out,
        resultText =
          "yaw ${pose.yaw.toInt()} pitch ${pose.pitch.toInt()} roll ${pose.roll.toInt()}",
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Draw the 3D head-pose axes centered on the face crop. */
  private fun drawAxis(face: Bitmap, hp: HeadPose): Bitmap {
    val out = face.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val cx = out.width / 2f
    val cy = out.height / 2f
    val size = out.width * 0.3f
    val p = Math.toRadians(hp.pitch.toDouble())
    val ya = Math.toRadians(-hp.yaw.toDouble())
    val r = Math.toRadians(hp.roll.toDouble())
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = out.width / 60f }
    paint.color = Color.rgb(255, 60, 60)
    canvas.drawLine(
      cx,
      cy,
      size * (cos(ya) * cos(r)).toFloat() + cx,
      size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(ya)).toFloat() + cy,
      paint,
    )
    paint.color = Color.rgb(60, 220, 90)
    canvas.drawLine(
      cx,
      cy,
      size * (-cos(ya) * sin(r)).toFloat() + cx,
      size * (cos(p) * cos(r) - sin(p) * sin(ya) * sin(r)).toFloat() + cy,
      paint,
    )
    paint.color = Color.rgb(70, 130, 255)
    canvas.drawLine(
      cx,
      cy,
      size * sin(ya).toFloat() + cx,
      size * (-cos(ya) * sin(p)).toFloat() + cy,
      paint,
    )
    return out
  }

  override fun onCleared() {
    super.onCleared()
    estimator?.close()
  }
}
