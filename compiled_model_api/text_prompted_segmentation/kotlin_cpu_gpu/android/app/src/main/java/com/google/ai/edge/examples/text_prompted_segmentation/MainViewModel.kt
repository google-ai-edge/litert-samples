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

package com.google.ai.edge.examples.text_prompted_segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
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
 * Owns the [ClipSeg] stack and exposes a single [UiState] for the screen. The two CLIP encoders and
 * the decoder reuse native buffers, so model creation and every segmentation run on one confined
 * worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    /** The prompt the screen starts with. */
    const val DEFAULT_PROMPT = "a cat"

    /** Sent to the text encoder when the user clears the prompt field. */
    private const val BLANK_PROMPT_FALLBACK = "object"

    /** Everything install_to_device.sh pushes; all of it must be present to build [ClipSeg]. */
    private val REQUIRED_FILES =
      listOf(
        "clipseg_vision_fp16.tflite",
        "clipseg_text_fp16.tflite",
        "clipseg_decoder.tflite",
        "token_embedding_f16.bin",
        "text_projection_f16.bin",
        "vocab.json",
        "merges.txt",
      )

    /** A mask value above this is blended; below it the pixel is left untouched. */
    private const val MASK_THRESHOLD = 0.1f

    /** Peak opacity of the red mask overlay. */
    private const val MASK_MAX_ALPHA = 0.6f

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var clipSeg: ClipSeg? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the models (from filesDir; pushed by install_to_device.sh).
    viewModelScope.launch(inferenceDispatcher) {
      val missing = REQUIRED_FILES.firstOrNull { !File(context.filesDir, it).exists() }
      if (missing != null) {
        val path = File(context.filesDir, missing).absolutePath
        val message = context.getString(R.string.error_model_missing, path)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        clipSeg = ClipSeg(context)
        _uiState.update { it.copy(isModelReady = true) }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Records the prompt the user is typing. */
  fun onPromptChange(prompt: String) {
    _uiState.update { it.copy(prompt = prompt) }
  }

  /** Loads a gallery image; it is segmented only when the user taps Segment. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val picked = context.decodeUriBitmap(uri)
        _uiState.update {
          it.copy(
            sourceImage = picked,
            resultImage = null,
            inferenceTimeMs = 0L,
            isMissingImage = false,
            errorMessage = null,
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_load_failed))
        }
      }
    }
  }

  /** Segments the picked image with the current prompt and blends the mask over it. */
  fun segment() {
    val state = _uiState.value
    val source =
      state.sourceImage
        ?: run {
          _uiState.update { it.copy(isMissingImage = true) }
          return
        }
    val prompt = state.prompt.ifBlank { BLANK_PROMPT_FALLBACK }
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      val clipSeg = clipSeg ?: return@launch
      try {
        val start = System.nanoTime()
        val mask = clipSeg.segment(source, prompt)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        val overlay = overlay(source, mask)
        _uiState.update {
          it.copy(
            isProcessing = false,
            resultImage = overlay,
            segmentedPrompt = prompt,
            inferenceTimeMs = elapsedMs,
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_segment_failed),
          )
        }
      }
    }
  }

  /** Blends a red mask over the image (mask is SIZE x SIZE, resized to the display bitmap). */
  private fun overlay(bitmap: Bitmap, mask: FloatArray): Bitmap {
    val width = bitmap.width
    val height = bitmap.height
    val out = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val pixels = IntArray(width * height)
    out.getPixels(pixels, 0, width, 0, 0, width, height)
    for (y in 0 until height) {
      for (x in 0 until width) {
        val maskX = (x * ClipSeg.SIZE / width).coerceIn(0, ClipSeg.SIZE - 1)
        val maskY = (y * ClipSeg.SIZE / height).coerceIn(0, ClipSeg.SIZE - 1)
        val maskValue = mask[maskY * ClipSeg.SIZE + maskX]
        if (maskValue > MASK_THRESHOLD) {
          val i = y * width + x
          val pixel = pixels[i]
          val alpha = (maskValue * MASK_MAX_ALPHA).coerceIn(0f, MASK_MAX_ALPHA)
          val red = (((pixel shr 16) and 0xFF) * (1 - alpha) + 255 * alpha).toInt()
          val green = (((pixel shr 8) and 0xFF) * (1 - alpha)).toInt()
          val blue = ((pixel and 0xFF) * (1 - alpha)).toInt()
          pixels[i] = Color.rgb(red, green, blue)
        }
      }
    }
    out.setPixels(pixels, 0, width, 0, 0, width, height)
    return out
  }

  override fun onCleared() {
    super.onCleared()
    clipSeg?.close()
  }
}
