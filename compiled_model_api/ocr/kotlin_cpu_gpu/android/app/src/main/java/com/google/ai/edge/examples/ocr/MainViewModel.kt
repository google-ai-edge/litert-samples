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

package com.google.ai.edge.examples.ocr

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
 * Owns the [PpocrDetector] + [PpocrRecognizer] and exposes a single [UiState] for the screen. Both
 * models reuse native buffers, so model creation and every OCR pass run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "test_image.png"

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var detector: PpocrDetector? = null
  private var recognizer: PpocrRecognizer? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load both models (from filesDir; pushed by install_to_device.sh) and read a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val missing =
        listOf(PpocrDetector.MODEL, PpocrRecognizer.MODEL)
          .map { File(context.filesDir, it) }
          .filter { !it.exists() }
      if (missing.isNotEmpty()) {
        _uiState.update {
          it.copy(
            errorMessage =
              "Model not found. Push it first with install_to_device.sh:\n" +
                missing.joinToString("\n") { file -> file.absolutePath }
          )
        }
        return@launch
      }
      try {
        detector = PpocrDetector(context)
        recognizer = PpocrRecognizer(context)
        runOcr(letterbox(context.decodeAssetBitmap(DEMO_IMAGE_ASSET)), warm = true)
      } catch (t: Throwable) {
        _uiState.update { it.copy(errorMessage = t.message ?: "Failed to load model") }
      }
    }
  }

  /** Runs OCR on a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        runOcr(letterbox(context.loadOrientedBitmap(uri)), warm = false)
      } catch (t: Throwable) {
        _uiState.update { it.copy(isProcessing = false, errorMessage = t.message ?: "OCR failed") }
      }
    }
  }

  /** Detect + recognize on a SIZE x SIZE bitmap, then annotate it and update the UI. */
  private fun runOcr(img: Bitmap, warm: Boolean) {
    val detector = detector ?: return
    val recognizer = recognizer ?: return
    val rgb = img.toRgbFloatArray()
    if (warm) detector.probMap(rgb) // warm up GPU shaders once
    val t0 = System.nanoTime()
    val boxes = detector.boxes(detector.probMap(rgb))
    val lines = ArrayList<Pair<PpocrDetector.Box, String>>()
    for (b in boxes) {
      val text = recognizer.recognize(cropResize(img, b))
      if (text.isNotBlank()) lines.add(b to text)
    }
    val elapsedMs = (System.nanoTime() - t0) / 1_000_000
    val annotated = drawBoxes(img, lines.map { it.first })
    val resultText =
      if (lines.isEmpty()) "(no text found)" else lines.joinToString("\n") { "• ${it.second}" }
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        resultImage = annotated,
        resultText = resultText,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  /**
   * Draws the detected boxes onto a mutable copy of the input bitmap. Adapted from the legacy
   * OverlayView.onDraw: the boxes are in SIZE space and we annotate the letterboxed SIZE x SIZE
   * input directly, so the model-input-to-source coordinate scale is identity (1:1).
   */
  private fun drawBoxes(img: Bitmap, boxes: List<PpocrDetector.Box>): Bitmap {
    val annotated = img.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(annotated)
    val stroke =
      Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 3f
      }
    for (b in boxes) {
      canvas.drawRect(b.x0.toFloat(), b.y0.toFloat(), b.x1.toFloat(), b.y1.toFloat(), stroke)
    }
    return annotated
  }

  /** Resize keeping aspect to fit SIZE, center on a white SIZE x SIZE canvas (letterbox). */
  private fun letterbox(src: Bitmap): Bitmap {
    val s =
      minOf(PpocrDetector.SIZE.toFloat() / src.width, PpocrDetector.SIZE.toFloat() / src.height)
    val nw = (src.width * s).toInt().coerceAtLeast(1)
    val nh = (src.height * s).toInt().coerceAtLeast(1)
    val out = Bitmap.createBitmap(PpocrDetector.SIZE, PpocrDetector.SIZE, Bitmap.Config.ARGB_8888)
    Canvas(out).apply {
      drawColor(Color.WHITE)
      drawBitmap(
        Bitmap.createScaledBitmap(src, nw, nh, true),
        (PpocrDetector.SIZE - nw) / 2f,
        (PpocrDetector.SIZE - nh) / 2f,
        null,
      )
    }
    return out
  }

  /** Crops a box from the SIZE x SIZE image and resizes+pads it to the recognizer's H x W input. */
  private fun cropResize(img: Bitmap, b: PpocrDetector.Box): FloatArray {
    val bw = b.x1 - b.x0 + 1
    val bh = b.y1 - b.y0 + 1
    val crop = Bitmap.createBitmap(img, b.x0, b.y0, bw, bh)
    val nw =
      minOf((PpocrRecognizer.H.toFloat() * bw / bh).toInt(), PpocrRecognizer.W).coerceAtLeast(1)
    val rz = Bitmap.createScaledBitmap(crop, nw, PpocrRecognizer.H, true)
    val px = IntArray(nw * PpocrRecognizer.H)
    rz.getPixels(px, 0, nw, 0, 0, nw, PpocrRecognizer.H)
    val out = FloatArray(PpocrRecognizer.H * PpocrRecognizer.W * 3)
    for (y in 0 until PpocrRecognizer.H) for (x in 0 until nw) {
      val p = px[y * nw + x]
      val o = (y * PpocrRecognizer.W + x) * 3
      out[o] = ((p shr 16) and 0xFF).toFloat()
      out[o + 1] = ((p shr 8) and 0xFF).toFloat()
      out[o + 2] = (p and 0xFF).toFloat()
    }
    return out
  }

  override fun onCleared() {
    super.onCleared()
    detector?.close()
    recognizer?.close()
  }
}
