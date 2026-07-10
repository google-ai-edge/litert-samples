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

package com.google.ai.edge.examples.image_tagging

import android.content.Context
import android.graphics.Bitmap
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
 * Owns the [RamTagger] and exposes a single [UiState] for the screen. [RamTagger] reuses native
 * input and output buffers across calls, so model creation and every tag run on one confined
 * worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val DEMO_IMAGE_ASSET = "test_image.jpg"

    /** The four graphs the hybrid pipeline loads from filesDir (pushed by install_to_device.sh). */
    private val MODEL_FILES =
      listOf(
        "ram_swin_s012_fp16.tflite",
        "ram_stage3_tail_fp16.tflite",
        "ram_taghead_fp16.tflite",
        "ram_reweight_fp16.tflite",
      )

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var tagger: RamTagger? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    // Load the graphs (from filesDir; pushed by install_to_device.sh) and tag a bundled image.
    viewModelScope.launch(inferenceDispatcher) {
      val missing = MODEL_FILES.map { File(context.filesDir, it) }.firstOrNull { !it.exists() }
      if (missing != null) {
        val message = context.getString(R.string.error_model_missing, missing.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        tagger = RamTagger(context)
        tag(context.decodeAssetBitmap(DEMO_IMAGE_ASSET))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Tags a gallery image picked by the user. */
  fun process(uri: Uri) {
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update { it.copy(isProcessing = true) }
      try {
        tag(context.loadOrientedBitmap(uri))
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(
            isProcessing = false,
            errorMessage = t.message ?: context.getString(R.string.error_tag_failed),
          )
        }
      }
    }
  }

  private fun tag(source: Bitmap) {
    val tagger = tagger ?: return
    val start = System.nanoTime()
    val tags = tagger.tag(source)
    val elapsedMs = (System.nanoTime() - start) / 1_000_000
    _uiState.update {
      UiState(
        isModelReady = true,
        isProcessing = false,
        sourceImage = source,
        tags = tags,
        inferenceTimeMs = elapsedMs,
      )
    }
  }

  override fun onCleared() {
    super.onCleared()
    tagger?.close()
  }
}
