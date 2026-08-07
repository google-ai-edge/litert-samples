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

package com.google.ai.edge.examples.face_alignment_3d

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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [TddfaLandmarks] helper and exposes a single [UiState] for the screen. [TddfaLandmarks]
 * reuses native input and output buffers across calls, so model creation and every landmark run
 * happen on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    /** Radius divisor and green fill used to draw each landmark dot (from the reference app). */
    private const val LANDMARK_RADIUS_DIVISOR = 220f
    private const val LANDMARK_MIN_RADIUS = 2f
    private val LANDMARK_COLOR = Color.rgb(0x00, 0xE5, 0x76)

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var tddfa: TddfaLandmarks? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from assets, inside TddfaLandmarks) and align a bundled sample face.
    viewModelScope.launch(inferenceDispatcher) {
      try {
        tddfa = TddfaLandmarks(context)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Aligns a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        detect(context.loadOrientedBitmap(uri))
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

  /** Runs the 3DMM regressor on [source], times it, and renders the 68-landmark overlay. */
  private fun detect(source: Bitmap) {
    val tddfa = tddfa ?: return
    val startNanos = System.nanoTime()
    val landmarks = tddfa.landmarks(source)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    if (landmarks == null) {
      _uiState.update {
        it.copy(
          isModelReady = true,
          isProcessing = false,
          faceDetected = false,
          resultImage = source,
          landmarkCount = 0,
        )
      }
      return
    }
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        faceDetected = true,
        resultImage = drawLandmarks(source, landmarks),
        landmarkCount = landmarks.size / 2,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /**
   * Draws each (x, y) landmark as a filled green dot onto a copy of [source]. Ported verbatim from
   * the reference activity: the dot radius scales with the larger image side.
   */
  private fun drawLandmarks(source: Bitmap, landmarks: FloatArray): Bitmap {
    val out = source.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val radius =
      (source.width.coerceAtLeast(source.height) / LANDMARK_RADIUS_DIVISOR)
        .coerceAtLeast(LANDMARK_MIN_RADIUS)
    val paint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = LANDMARK_COLOR
        style = Paint.Style.FILL
      }
    for (n in 0 until landmarks.size / 2) {
      canvas.drawCircle(landmarks[n * 2], landmarks[n * 2 + 1], radius, paint)
    }
    return out
  }

  override fun onCleared() {
    super.onCleared()
    tddfa?.close()
  }
}
