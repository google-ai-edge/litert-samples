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

package com.google.ai.edge.examples.image_matching

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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
 * Owns the [XFeatMatcher] and exposes a single [UiState] for the screen. [XFeatMatcher] reuses
 * native input and output buffers across calls, so model creation and every extract/match run on
 * one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    const val FIRST_IMAGE_SLOT = 1
    const val SECOND_IMAGE_SLOT = 2

    private const val MODEL_FILE = "xfeat_fp16.tflite"

    /**
     * The result bitmap draws the two 640x480 model-space images side by side at native
     * resolution. With each half sized to [XFeatMatcher.W] x [XFeatMatcher.H], the match-line
     * scale factors resolve to sx = sy = 1, so the ported stroke width (2.5f) and endpoint radius
     * (3.5f) keep the exact proportions the old MatchView drew on screen. Compose then fits this
     * whole bitmap to the display, scaling image and lines together.
     */
    private const val RESULT_IMAGE_WIDTH = XFeatMatcher.W * 2
    private const val RESULT_IMAGE_HEIGHT = XFeatMatcher.H

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var matcher: XFeatMatcher? = null
  private var firstImage: Bitmap? = null
  private var secondImage: Bitmap? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model from filesDir (pushed by install_to_device.sh).
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        matcher = XFeatMatcher(context)
        _uiState.update { it.copy(isModelReady = true) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /**
   * Stores the image picked into [slot] (1 or 2) and, once both images are present, runs matching.
   */
  fun onImagePicked(slot: Int, uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val bitmap = context.decodeUriBitmap(uri)
        if (slot == FIRST_IMAGE_SLOT) firstImage = bitmap else secondImage = bitmap
        _uiState.update {
          it.copy(
            lastPickedSlot = slot,
            lastPickedWidth = bitmap.width,
            lastPickedHeight = bitmap.height,
            resultImage = null,
            errorMessage = null,
          )
        }
        val a = firstImage
        val b = secondImage
        if (a != null && b != null) match(a, b)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_pick_failed))
        }
      }
    }
  }

  /** Extracts features from both images on the GPU and matches them (host side). */
  private fun match(a: Bitmap, b: Bitmap) {
    val matcher = matcher ?: return
    _uiState.update { it.copy(isProcessing = true) }
    val startNanos = System.nanoTime()
    val featuresA = matcher.extract(matcher.preprocess(a))
    val featuresB = matcher.extract(matcher.preprocess(b))
    val matches = matcher.match(featuresA, featuresB)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    val scaledA = Bitmap.createScaledBitmap(a, XFeatMatcher.W, XFeatMatcher.H, true)
    val scaledB = Bitmap.createScaledBitmap(b, XFeatMatcher.W, XFeatMatcher.H, true)
    val result = renderMatchImage(scaledA, scaledB, matches)
    _uiState.update {
      it.copy(
        isProcessing = false,
        resultImage = result,
        matchCount = matches.size,
        keypointCountA = featuresA.xs.size,
        keypointCountB = featuresB.xs.size,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /**
   * Draws the two 640x480 images side by side and overlays the match lines. This is the old
   * MatchView.onDraw math, unchanged, targeting a bitmap instead of the on-screen canvas.
   */
  private fun renderMatchImage(
    imageA: Bitmap,
    imageB: Bitmap,
    matches: List<XFeatMatcher.Match>,
  ): Bitmap {
    val output =
      Bitmap.createBitmap(RESULT_IMAGE_WIDTH, RESULT_IMAGE_HEIGHT, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    val half = RESULT_IMAGE_WIDTH / 2f
    val hgt = RESULT_IMAGE_HEIGHT.toFloat()
    canvas.drawBitmap(imageA, null, RectF(0f, 0f, half, hgt), null)
    canvas.drawBitmap(imageB, null, RectF(half, 0f, RESULT_IMAGE_WIDTH.toFloat(), hgt), null)
    val sx = half / XFeatMatcher.W
    val sy = hgt / XFeatMatcher.H
    paint.strokeWidth = 2.5f
    for (m in matches) {
      // green (high sim) -> yellow (borderline)
      val t =
        ((m.sim - XFeatMatcher.MIN_COSSIM) / (1f - XFeatMatcher.MIN_COSSIM)).coerceIn(0f, 1f)
      paint.color = Color.argb(200, (255 * (1 - t)).toInt(), 220, 40)
      canvas.drawLine(m.x0 * sx, m.y0 * sy, half + m.x1 * sx, m.y1 * sy, paint)
      canvas.drawCircle(m.x0 * sx, m.y0 * sy, 3.5f, paint)
      canvas.drawCircle(half + m.x1 * sx, m.y1 * sy, 3.5f, paint)
    }
    return output
  }

  override fun onCleared() {
    super.onCleared()
    matcher?.close()
  }
}
