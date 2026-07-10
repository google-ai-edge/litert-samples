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

package com.google.ai.edge.examples.saliency

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import kotlin.math.abs
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [SaliencyPredictor] and exposes a single [UiState] for the screen. [SaliencyPredictor]
 * reuses native input and output buffers across calls, so model creation and every prediction run
 * on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "unisal_fp16.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var predictor: SaliencyPredictor? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model from filesDir (pushed by install_to_device.sh) and predict a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        predictor = SaliencyPredictor(context)
        predict(squareResize(context.decodeAssetBitmap(DEMO_IMAGE_ASSET)), warmUp = true)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Predicts saliency for a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        predict(squareResize(context.loadOrientedBitmap(uri)), warmUp = false)
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_predict_failed),
          )
        }
      }
    }
  }

  /**
   * Runs the model on a square [source] bitmap and publishes the jet-heatmap overlay. When [warmUp]
   * is true an extra untimed prediction primes the GPU delegate before the measured run.
   */
  private fun predict(source: Bitmap, warmUp: Boolean) {
    val predictor = predictor ?: return
    val rgb = bitmapToRgb(source)
    if (warmUp) {
      predictor.predict(rgb)
    }
    val startNs = System.nanoTime()
    val saliency = predictor.predict(rgb)
    val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
    val heatmap = overlay(source, saliency)
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = source,
        resultImage = heatmap,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /** Center-crops [src] to a square and resizes it to the model's input resolution. */
  private fun squareResize(src: Bitmap): Bitmap {
    val s = min(src.width, src.height)
    val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
    return Bitmap.createScaledBitmap(crop, SaliencyPredictor.SIZE, SaliencyPredictor.SIZE, true)
  }

  /** Flattens a bitmap to a row-major RGB float array in [0, 255], as the model expects. */
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

  /** Blends a jet heatmap of the saliency over the image. */
  private fun overlay(bm: Bitmap, sal: FloatArray): Bitmap {
    val s = SaliencyPredictor.SIZE
    val src =
      if (bm.width == s && bm.height == s) bm else Bitmap.createScaledBitmap(bm, s, s, true)
    val px = IntArray(s * s)
    src.getPixels(px, 0, s, 0, 0, s, s)
    val out = IntArray(s * s)
    for (i in px.indices) {
      val v = sal[i]
      val (hr, hg, hb) = jet(v)
      val a = 0.55f * v + 0.15f // more opaque where salient
      val p = px[i]
      val r = ((p shr 16) and 0xFF) * (1 - a) + hr * a
      val g = ((p shr 8) and 0xFF) * (1 - a) + hg * a
      val b2 = (p and 0xFF) * (1 - a) + hb * a
      out[i] =
        Color.rgb(
          r.toInt().coerceIn(0, 255),
          g.toInt().coerceIn(0, 255),
          b2.toInt().coerceIn(0, 255),
        )
    }
    return Bitmap.createBitmap(out, s, s, Bitmap.Config.ARGB_8888)
  }

  /** Jet colormap: [0, 1] -> (r, g, b) in [0, 255]. */
  private fun jet(x: Float): Triple<Float, Float, Float> {
    val v = x.coerceIn(0f, 1f)
    fun clamp(a: Float) = (a.coerceIn(0f, 1f)) * 255f
    val r = clamp(1.5f - abs(4f * v - 3f))
    val g = clamp(1.5f - abs(4f * v - 2f))
    val b = clamp(1.5f - abs(4f * v - 1f))
    return Triple(r, g, b)
  }

  override fun onCleared() {
    super.onCleared()
    predictor?.close()
  }
}
