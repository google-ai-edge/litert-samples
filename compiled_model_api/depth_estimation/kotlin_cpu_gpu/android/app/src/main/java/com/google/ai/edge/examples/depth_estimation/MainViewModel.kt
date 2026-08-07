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

package com.google.ai.edge.examples.depth_estimation

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

class MainViewModel(private val depthEstimationHelper: DepthEstimationHelper) : ViewModel() {
  companion object {
    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          val depthEstimationHelper = DepthEstimationHelper(context)
          return if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            MainViewModel(depthEstimationHelper) as T
          } else {
            throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
          }
        }
      }
  }

  private var currentModel = DepthEstimationHelper.Model.MidasSmall
  private var currentAccelerator = DepthEstimationHelper.AcceleratorEnum.GPU

  init {
    viewModelScope.launch { depthEstimationHelper.initEstimator(currentModel, currentAccelerator) }
  }

  private var estimateJob: Job? = null

  // Latest colorized depth map paired with its inference time.
  private val depthUiShareFlow =
    MutableStateFlow<Pair<Bitmap?, Long>>(Pair(null, 0L)).also { flow ->
      viewModelScope.launch {
        depthEstimationHelper.depth
          .map { Pair(it.overlay, it.inferenceTime) }
          .collect { flow.emit(it) }
      }
    }

  private val mediaUri = MutableStateFlow<Uri>(Uri.EMPTY)

  private val errorMessage =
    MutableStateFlow<Throwable?>(null).also {
      viewModelScope.launch { depthEstimationHelper.error.collect(it) }
    }

  private val lensFacing = MutableStateFlow(CameraSelector.LENS_FACING_BACK)

  val uiState: StateFlow<UiState> =
    combine(mediaUri, depthUiShareFlow, errorMessage, lensFacing) {
        uri, depthPair, error, lensFace ->
        UiState(
          mediaUri = uri,
          overlay = depthPair.first,
          inferenceTime = depthPair.second,
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
   * Estimate depth on a camera frame.
   *
   * The frame is rotated upright first so the depth map aligns with the on-screen preview.
   *
   * @param imageProxy contains the camera bitmap and its rotation degrees.
   */
  fun estimate(imageProxy: ImageProxy) {
    estimateJob =
      viewModelScope.launch {
        val bitmap = imageProxy.toBitmap().rotate(imageProxy.imageInfo.rotationDegrees)
        depthEstimationHelper.estimate(bitmap)
        imageProxy.close()
      }
  }

  /**
   * Estimate depth on a gallery image or video frame.
   *
   * @param bitmap source image; copied to [Bitmap.Config.ARGB_8888] before inference.
   */
  fun estimate(bitmap: Bitmap) {
    estimateJob =
      viewModelScope.launch {
        val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        depthEstimationHelper.estimate(argbBitmap)
      }
  }

  /** Stop the current estimation and clear the overlay. */
  fun stopEstimate() {
    viewModelScope.launch {
      estimateJob?.cancel()
      depthUiShareFlow.emit(Pair(null, 0L))
    }
  }

  /** Update display media uri */
  fun updateMediaUri(uri: Uri) {
    if (uri != mediaUri.value || uri.toString().contains("video")) {
      stopEstimate()
    }
    mediaUri.update { uri }
  }

  /** Set Accelerator for DepthEstimationHelper (CPU/GPU) */
  fun setAccelerator(acceleratorEnum: DepthEstimationHelper.AcceleratorEnum) {
    currentAccelerator = acceleratorEnum
    viewModelScope.launch { depthEstimationHelper.initEstimator(currentModel, currentAccelerator) }
  }

  /** Set the depth Model (MiDaS-small / Depth Anything 3) */
  fun setModel(model: DepthEstimationHelper.Model) {
    currentModel = model
    viewModelScope.launch { depthEstimationHelper.initEstimator(currentModel, currentAccelerator) }
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
