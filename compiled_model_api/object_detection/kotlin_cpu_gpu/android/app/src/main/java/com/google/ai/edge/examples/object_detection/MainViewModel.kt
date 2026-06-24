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

package com.google.ai.edge.examples.object_detection

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
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val detectionHelper: ObjectDetectionHelper) : ViewModel() {
  companion object {
    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          val detectionHelper = ObjectDetectionHelper(context)
          return if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            MainViewModel(detectionHelper) as T
          } else {
            throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
          }
        }
      }
  }

  init {
    viewModelScope.launch { detectionHelper.initDetector() }
  }

  private var detectJob: Job? = null

  private val emptyResult = ObjectDetectionHelper.DetectionResult(emptyList(), 0, 0, 0L)

  // Latest detected objects (boxes + source size + inference time).
  private val detectionShareFlow =
    MutableStateFlow(emptyResult).also { flow ->
      viewModelScope.launch { detectionHelper.detections.collect { flow.emit(it) } }
    }

  private val mediaUri = MutableStateFlow<Uri>(Uri.EMPTY)

  private val errorMessage =
    MutableStateFlow<Throwable?>(null).also {
      viewModelScope.launch { detectionHelper.error.collect(it) }
    }

  private val lensFacing = MutableStateFlow(CameraSelector.LENS_FACING_BACK)

  val uiState: StateFlow<UiState> =
    combine(mediaUri, detectionShareFlow, errorMessage, lensFacing) { uri, result, error, lensFace ->
        UiState(
          mediaUri = uri,
          detections = result.detections,
          sourceWidth = result.sourceWidth,
          sourceHeight = result.sourceHeight,
          inferenceTime = result.inferenceTime,
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
   * Detect objects in a camera frame.
   *
   * The frame is rotated upright first so the boxes align with the on-screen preview.
   *
   * @param imageProxy contains the camera bitmap and its rotation degrees.
   */
  fun detect(imageProxy: ImageProxy) {
    detectJob =
      viewModelScope.launch {
        val bitmap = imageProxy.toBitmap().rotate(imageProxy.imageInfo.rotationDegrees)
        detectionHelper.detect(bitmap)
        imageProxy.close()
      }
  }

  /**
   * Detect objects in a gallery image or video frame.
   *
   * @param bitmap source image; copied to [Bitmap.Config.ARGB_8888] before inference.
   */
  fun detect(bitmap: Bitmap) {
    detectJob =
      viewModelScope.launch {
        val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        detectionHelper.detect(argbBitmap)
      }
  }

  /** Stop the current detection and clear the boxes. */
  fun stopDetect() {
    viewModelScope.launch {
      detectJob?.cancel()
      detectionShareFlow.emit(emptyResult)
    }
  }

  /** Update display media uri */
  fun updateMediaUri(uri: Uri) {
    if (uri != mediaUri.value || uri.toString().contains("video")) {
      stopDetect()
    }
    mediaUri.update { uri }
  }

  /** Set Accelerator for ObjectDetectionHelper (CPU/GPU) */
  fun setAccelerator(acceleratorEnum: ObjectDetectionHelper.AcceleratorEnum) {
    viewModelScope.launch { detectionHelper.initDetector(acceleratorEnum) }
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
