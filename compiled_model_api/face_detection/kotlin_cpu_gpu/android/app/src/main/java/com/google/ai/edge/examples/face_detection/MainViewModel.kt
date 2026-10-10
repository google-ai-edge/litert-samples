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

package com.google.ai.edge.examples.face_detection

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
 * Owns the [FaceDetector] and exposes a single [UiState] for the screen. [FaceDetector] reuses
 * native input and output buffers across calls, so model creation and every detection run on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "yunet_fp16.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    // Overlay styling ported verbatim from the old FaceView.onDraw.
    private const val BOX_STROKE_WIDTH = 4f
    private const val LANDMARK_RADIUS = 4f
    private const val LANDMARK_COUNT = 5

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var detector: FaceDetector? = null

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
        detector = FaceDetector(context)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET), warm = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Detects faces in a gallery image picked by the user. */
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
   * Letterboxes [source] to the model's square input, runs the detector, and publishes an annotated
   * result. When [warm] is true an extra untimed inference precedes the timed one, so the reported
   * time excludes first-run/GPU warm-up cost.
   */
  private fun detect(source: Bitmap, warm: Boolean) {
    val detector = detector ?: return
    val square = source.squareResize()
    val rgb = square.toRgbFloatArray()
    if (warm) {
      detector.detect(rgb)
    }
    val startNs = System.nanoTime()
    val faces = detector.detect(rgb)
    val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = drawFaces(square, faces),
        inferenceTimeMs = elapsedMs,
        faceCount = faces.size,
      )
    }
  }

  /**
   * Draws boxes and 5 landmarks onto a copy of the SIZE×SIZE letterboxed [square]. Ported from the
   * old FaceView.onDraw: because we annotate the model-space bitmap at native resolution, the
   * fit-scale/offset collapse to identity (scale = 1, offsetX = offsetY = 0), so face coordinates
   * (already in SIZE×SIZE space) are drawn directly.
   */
  private fun drawFaces(square: Bitmap, faces: List<FaceDetector.Face>): Bitmap {
    val annotated = square.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(annotated)
    val boxPaint =
      Paint().apply {
        color = Color.rgb(0, 230, 0)
        style = Paint.Style.STROKE
        strokeWidth = BOX_STROKE_WIDTH
        isAntiAlias = true
      }
    val landmarkPaint =
      Paint().apply {
        color = Color.rgb(255, 40, 40)
        isAntiAlias = true
      }
    for (face in faces) {
      canvas.drawRect(face.x1, face.y1, face.x2, face.y2, boxPaint)
      for (j in 0 until LANDMARK_COUNT) {
        canvas.drawCircle(
          face.landmarks[2 * j],
          face.landmarks[2 * j + 1],
          LANDMARK_RADIUS,
          landmarkPaint,
        )
      }
    }
    return annotated
  }

  override fun onCleared() {
    super.onCleared()
    detector?.close()
  }
}
