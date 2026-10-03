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

package com.google.ai.edge.examples.inpainting

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
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
 * Owns the [MiganInpainter] and exposes a single [UiState] for the screen. The inpainter reuses
 * native input and output buffers, so model creation and every erase run on one confined worker.
 *
 * The user paints strokes over the region to remove; [erase] rasterizes them into the binary mask
 * MI-GAN expects (0 = erase, 1 = keep) at the model's native resolution.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val MODEL_FILE = "migan_fp16.tflite"
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    /** Brush radius in image space, as a fraction of the model's input size. */
    private const val BRUSH_RADIUS = MiganInpainter.SIZE * 0.06f

    /** A mask pixel counts as painted once its blue channel crosses this value. */
    private const val PAINTED_THRESHOLD = 127

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var inpainter: MiganInpainter? = null

  /** The last image loaded from assets or the gallery, restored by [reset]. */
  private var originalImage: Bitmap? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the model (from filesDir; pushed by install_to_device.sh) and show a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, MODEL_FILE)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        inpainter = MiganInpainter(context)
        val decoded = context.decodeAssetBitmap(DEMO_IMAGE_ASSET)
        val demo = decoded.centerCropToSquare(MiganInpainter.SIZE)
        originalImage = demo
        _uiState.update { UiState(isModelReady = true, image = demo) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Loads a gallery image, replacing the working image and clearing any painted strokes. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val picked = context.loadOrientedBitmap(uri).centerCropToSquare(MiganInpainter.SIZE)
        originalImage = picked
        _uiState.update { UiState(isModelReady = true, image = picked) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_load_failed))
        }
      }
    }
  }

  /** Begins a new stroke at the given point, in SIZE x SIZE image coordinates. */
  fun startStroke(x: Float, y: Float) {
    _uiState.update {
      it.copy(strokes = it.strokes + Stroke(listOf(StrokePoint(x, y))), isMissingStrokes = false)
    }
  }

  /** Appends a point to the stroke currently being painted. */
  fun extendStroke(x: Float, y: Float) {
    _uiState.update { state ->
      val current = state.strokes.lastOrNull() ?: return@update state
      val extended = Stroke(current.points + StrokePoint(x, y))
      state.copy(strokes = state.strokes.dropLast(1) + extended)
    }
  }

  /** Discards the painted strokes and restores the image as it was last loaded. */
  fun reset() {
    val original = originalImage ?: return
    _uiState.update { UiState(isModelReady = true, image = original) }
  }

  /** Rasterizes the painted strokes into a mask and inpaints the masked region on the GPU. */
  fun erase() {
    val state = _uiState.value
    val image = state.image ?: return
    if (state.strokes.isEmpty()) {
      _uiState.update { it.copy(isMissingStrokes = true) }
      return
    }
    val mask = buildMask(state.strokes)
    val rgb = image.toRgbFloatArray()
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      val inpainter = inpainter ?: return@launch
      try {
        val start = System.nanoTime()
        val result = inpainter.inpaint(rgb, mask)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        _uiState.update {
          UiState(isModelReady = true, image = result, inferenceTimeMs = elapsedMs)
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_erase_failed),
          )
        }
      }
    }
  }

  /**
   * Rasterizes [strokes] to a SIZE x SIZE mask: 0 = erase, 1 = keep. Anti-aliasing is off so the
   * mask stays strictly binary.
   */
  private fun buildMask(strokes: List<Stroke>): FloatArray {
    val size = MiganInpainter.SIZE
    val maskBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(maskBitmap)
    canvas.drawColor(Color.BLACK)
    val brush =
      Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = BRUSH_RADIUS * 2
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = false
      }
    for (stroke in strokes) {
      canvas.drawPath(stroke.toPath(), brush)
    }
    val pixels = IntArray(size * size)
    maskBitmap.getPixels(pixels, 0, size, 0, 0, size, size)
    val mask = FloatArray(size * size) { if ((pixels[it] and 0xFF) > PAINTED_THRESHOLD) 0f else 1f }
    maskBitmap.recycle()
    return mask
  }

  private fun Stroke.toPath(): Path {
    val path = Path()
    val first = points.first()
    path.moveTo(first.x, first.y)
    for (point in points.drop(1)) {
      path.lineTo(point.x, point.y)
    }
    return path
  }

  override fun onCleared() {
    super.onCleared()
    inpainter?.close()
  }
}
