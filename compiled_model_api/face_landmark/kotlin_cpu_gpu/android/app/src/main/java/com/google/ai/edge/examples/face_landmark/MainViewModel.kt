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

package com.google.ai.edge.examples.face_landmark

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
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [RtmFaceEstimator] and exposes a single [UiState] for the screen. [RtmFaceEstimator]
 * reuses native input and output buffers across calls, so model creation and every estimate run on
 * one confined worker.
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

  private var estimator: RtmFaceEstimator? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  /** WFLW 98-landmark groups: (indices, closedLoop). */
  private val groups =
    arrayOf(
      (0..32).toList() to false, // face contour / jaw
      (33..41).toList() to false, // left eyebrow
      (42..50).toList() to false, // right eyebrow
      (51..54).toList() to false, // nose bridge
      (55..59).toList() to false, // nose bottom
      (60..67).toList() to true, // left eye
      (68..75).toList() to true, // right eye
      (76..87).toList() to true, // outer lips
      (88..95).toList() to true, // inner lips
    )

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and estimate a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, RtmFaceEstimator.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        estimator = RtmFaceEstimator(context)
        estimate(cropSquare(context.decodeAssetBitmap(DEMO_IMAGE_ASSET)), warm = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Estimates landmarks for a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        estimate(cropSquare(context.loadOrientedBitmap(uri)), warm = false)
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
   * Runs the estimator on a square face [crop] and publishes the annotated mesh. When [warm] is
   * true a discarded warm-up run precedes the timed run so the reported latency excludes one-time
   * GPU shader compilation.
   */
  private fun estimate(crop: Bitmap, warm: Boolean) {
    val net = estimator ?: return
    val rgb = bitmapToRgb(crop)
    if (warm) {
      net.estimate(rgb)
    }
    val startNanos = System.nanoTime()
    val points = net.estimate(rgb)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = drawMesh(crop, points),
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /**
   * Draws the 98-landmark face mesh onto a mutable copy of [crop]: green edges within each landmark
   * group (closing the eye and lip loops) plus a red dot per point. Coordinates come back in crop
   * pixels, so they map straight onto the copy at native resolution.
   */
  private fun drawMesh(crop: Bitmap, points: List<RtmFaceEstimator.Point>): Bitmap {
    val output = crop.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(output)
    val line =
      Paint().apply {
        color = Color.rgb(0, 230, 0)
        strokeWidth = 2.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
      }
    val dot =
      Paint().apply {
        color = Color.rgb(255, 40, 40)
        isAntiAlias = true
      }
    fun px(i: Int) = points[i].x
    fun py(i: Int) = points[i].y
    for ((ids, closed) in groups) {
      for (j in 0 until ids.size - 1) {
        canvas.drawLine(px(ids[j]), py(ids[j]), px(ids[j + 1]), py(ids[j + 1]), line)
      }
      if (closed && ids.size > 1) {
        canvas.drawLine(px(ids.last()), py(ids.last()), px(ids.first()), py(ids.first()), line)
      }
    }
    for (i in points.indices) {
      canvas.drawCircle(px(i), py(i), 2.5f, dot)
    }
    return output
  }

  /** Center-crops [src] to a square and scales it to the model's input resolution. */
  private fun cropSquare(src: Bitmap): Bitmap {
    val s = min(src.width, src.height)
    val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    return Bitmap.createScaledBitmap(crop, RtmFaceEstimator.W, RtmFaceEstimator.H, true)
  }

  /** Flattens [bm] to a row-major RGB [0,255] float array (the estimator's expected input). */
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

  override fun onCleared() {
    super.onCleared()
    estimator?.close()
  }
}
