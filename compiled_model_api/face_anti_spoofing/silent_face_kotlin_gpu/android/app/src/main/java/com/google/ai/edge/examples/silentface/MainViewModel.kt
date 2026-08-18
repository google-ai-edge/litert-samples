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

package com.google.ai.edge.examples.silentface

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
 * Owns the [LivenessDetector] and exposes a single [UiState] for the screen. The detector reuses
 * native buffers, so both model creation and every detect run on one confined worker.
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

  private var detector: LivenessDetector? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (1.85 MB, bundled in assets) and score a bundled face photo at launch.
    viewModelScope.launch(inferenceDispatcher) {
      try {
        detector = LivenessDetector(context)
        detect(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs liveness detection on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        detect(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(isProcessing = false, errorMessage = t.message ?: "Detection failed")
        }
      }
    }
  }

  private fun detect(source: Bitmap) {
    val detector = detector ?: return
    val (p, ms) = detector.detect(source)
    val live = p[1] >= p[0] && p[1] >= p[2]
    val out = annotate(source, p, live)
    val detail =
      "Silent-Face liveness  ·  CompiledModel GPU  ·  ${ms} ms  ·  " +
        "${if (live) "LIVE" else "SPOOF"} (live ${(p[1] * 100).toInt()}%)"
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = out,
        resultText = detail,
        inferenceTimeMs = ms,
      )
    }
  }

  /** Draws a colored border (green = live, red = spoof) and verdict onto a copy of the input. */
  private fun annotate(image: Bitmap, p: FloatArray, live: Boolean): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val c = Canvas(out)
    val col = if (live) Color.rgb(50, 220, 100) else Color.rgb(240, 70, 70)
    val box =
      Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = out.width / 90f
        color = col
      }
    c.drawRect(4f, 4f, out.width - 4f, out.height - 4f, box)
    val tp =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = col
        textSize = out.width / 8f
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
      }
    c.drawText(if (live) "LIVE" else "SPOOF", 20f, tp.textSize + 20f, tp)
    return out
  }

  override fun onCleared() {
    super.onCleared()
    detector?.close()
  }
}
