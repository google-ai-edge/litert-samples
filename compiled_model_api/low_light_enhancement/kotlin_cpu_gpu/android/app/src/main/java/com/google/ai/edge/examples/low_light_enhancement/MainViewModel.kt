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

package com.google.ai.edge.examples.low_light_enhancement

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val enhancementHelper: EnhancementHelper) : ViewModel() {
  companion object {
    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          val enhancementHelper = EnhancementHelper(context)
          return if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            MainViewModel(enhancementHelper) as T
          } else {
            throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
          }
        }
      }
  }

  init {
    viewModelScope.launch { enhancementHelper.initEnhancer() }
  }

  private var enhanceJob: Job? = null

  // Latest enhanced image paired with its inference time.
  private val enhancedUiShareFlow =
    MutableStateFlow<Pair<Bitmap?, Long>>(Pair(null, 0L)).also { flow ->
      viewModelScope.launch {
        enhancementHelper.enhanced
          .map { Pair(it.overlay, it.inferenceTime) }
          .collect { flow.emit(it) }
      }
    }

  private val mediaUri = MutableStateFlow<Uri>(Uri.EMPTY)

  private val errorMessage =
    MutableStateFlow<Throwable?>(null).also {
      viewModelScope.launch { enhancementHelper.error.collect(it) }
    }

  private val lensFacing = MutableStateFlow(CameraSelector.LENS_FACING_BACK)

  val uiState: StateFlow<UiState> =
    combine(mediaUri, enhancedUiShareFlow, errorMessage, lensFacing) { uri, enhancedPair, error, lensFace ->
        UiState(
          mediaUri = uri,
          overlay = enhancedPair.first,
          inferenceTime = enhancedPair.second,
          errorMessage = error?.message,
          lensFacing = lensFace,
        )
      }
      .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

  fun flipCamera() {
    val newFacing =
      if (lensFacing.value == CameraSelector.LENS_FACING_BACK) {
        CameraSelector.LENS_FACING_FRONT
      } else {
        CameraSelector.LENS_FACING_BACK
      }

    lensFacing.update { newFacing }
  }

  /**
   * Enhance a camera frame.
   *
   * The frame is rotated upright first so the result aligns with the on-screen preview.
   *
   * @param imageProxy contains the camera bitmap and its rotation degrees.
   */
  fun enhance(imageProxy: ImageProxy) {
    enhanceJob =
      viewModelScope.launch {
        val bitmap = imageProxy.toBitmap().rotate(imageProxy.imageInfo.rotationDegrees)
        enhancementHelper.enhance(bitmap)
        imageProxy.close()
      }
  }

  /**
   * Enhance a gallery image or video frame.
   *
   * @param bitmap source image; copied to [Bitmap.Config.ARGB_8888] before inference.
   */
  fun enhance(bitmap: Bitmap) {
    enhanceJob =
      viewModelScope.launch {
        val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        enhancementHelper.enhance(argbBitmap)
      }
  }

  /** Stop the current enhancement and clear the output. */
  fun stopEnhance() {
    viewModelScope.launch {
      enhanceJob?.cancel()
      enhancedUiShareFlow.emit(Pair(null, 0L))
    }
  }

  /** Update display media uri */
  fun updateMediaUri(uri: Uri) {
    if (uri != mediaUri.value || uri.toString().contains("video")) {
      stopEnhance()
    }
    mediaUri.update { uri }
  }

  /** Set Accelerator for EnhancementHelper (CPU/GPU) */
  fun setAccelerator(acceleratorEnum: EnhancementHelper.AcceleratorEnum) {
    viewModelScope.launch { enhancementHelper.initEnhancer(acceleratorEnum) }
  }

  /** Clear error message after it has been consumed */
  fun errorMessageShown() {
    errorMessage.update { null }
  }

  private fun Bitmap.rotate(degrees: Int): Bitmap {
    if (degrees == 0) return this
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
  }
}
