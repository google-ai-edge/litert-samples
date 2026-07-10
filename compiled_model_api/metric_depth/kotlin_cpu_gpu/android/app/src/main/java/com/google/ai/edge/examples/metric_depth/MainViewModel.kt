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

package com.google.ai.edge.examples.metric_depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [MetricDepth] model and exposes a single [UiState] for the screen. [MetricDepth] reuses
 * native input and output buffers across calls, so model creation and every depth run are confined
 * to a single worker thread.
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

  private var metricDepth: MetricDepth? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (pushed to filesDir by install_to_device.sh) and warm up on a demo image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MetricDepth.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        metricDepth = MetricDepth(context)
        estimate(squareResize(context.decodeAssetBitmap(DEMO_IMAGE_ASSET)), warmUp = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Estimates metric depth on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        estimate(squareResize(context.loadOrientedBitmap(uri)), warmUp = false)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_depth_failed),
          )
        }
      }
    }
  }

  /**
   * Runs the model on a SIZE×SIZE [source] bitmap and publishes the colorized depth map plus the
   * near/far metric range. When [warmUp] is true an extra untimed pass primes the GPU shaders.
   */
  private fun estimate(source: Bitmap, warmUp: Boolean) {
    val net = metricDepth ?: return
    val rgb = bitmapToRgb(source)
    if (warmUp) {
      net.depth(rgb) // warm up GPU shaders once
    }
    val startNanos = System.nanoTime()
    val depth = net.depth(rgb)
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    // Robust min/max over the 2nd..98th percentile for the colormap and labels.
    val sorted = depth.clone().also { it.sort() }
    val nearMeters = sorted[(sorted.size * 0.02f).toInt()]
    val farMeters = sorted[(sorted.size * 0.98f).toInt()]
    val depthMap = colorize(depth, MetricDepth.SIZE, nearMeters, farMeters)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = source,
        depthImage = depthMap,
        inferenceTimeMs = elapsedMs,
        nearMeters = nearMeters,
        farMeters = farMeters,
      )
    }
  }

  /**
   * Center-crops [src] to square, then resizes to SIZE×SIZE (preserves local geometry; no letterbox
   * padding).
   */
  private fun squareResize(src: Bitmap): Bitmap {
    val s = min(src.width, src.height)
    val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    return Bitmap.createScaledBitmap(crop, MetricDepth.SIZE, MetricDepth.SIZE, true)
  }

  /** Flattens a bitmap to row-major RGB floats in the [0,255] range the model expects. */
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

  /** Maps depth (meters) to a Turbo colormap bitmap. Near = warm, far = cool. */
  private fun colorize(depth: FloatArray, size: Int, lo: Float, hi: Float): Bitmap {
    val px = IntArray(size * size)
    val span = max(hi - lo, 1e-3f)
    for (i in depth.indices) {
      val t = ((depth[i] - lo) / span).coerceIn(0f, 1f)
      px[i] = turbo(1f - t) // invert so near objects are warm/bright
    }
    return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
  }

  /** Google "Turbo" colormap approximation, t in [0,1]. */
  private fun turbo(t: Float): Int {
    val r =
      (34.61 +
          t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))))
        .toInt()
    val g =
      (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (4520.0 + t * -1894.0))))).toInt()
    val b =
      (27.2 +
          t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))))
        .toInt()
    return Color.rgb(r.coerceIn(0, 255), g.coerceIn(0, 255), b.coerceIn(0, 255))
  }

  override fun onCleared() {
    super.onCleared()
    metricDepth?.close()
  }
}
