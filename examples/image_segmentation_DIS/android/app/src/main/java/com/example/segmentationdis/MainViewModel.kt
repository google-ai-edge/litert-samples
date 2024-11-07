/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.example.segmentationdis

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val imageSegmentationHelper: ImageSegmentationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                // To apply object detection, we use our ObjectDetectorHelper class,
                // which abstracts away the specifics of using MediaPipe  for object
                // detection from the UI elements of the app
                val imageSegmentationHelper = ImageSegmentationHelper(context)
                return MainViewModel(imageSegmentationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch {
            imageSegmentationHelper.initClassifier()
        }
        viewModelScope.launch {
            imageSegmentationHelper.segmentation.map { OverlayInfo(bitmap = it.bitmap) to it.inferenceTime }
                .collect { info ->
                    _uiState.update {
                        if (it.currentTab == Tab.Camera) {
                            _uiState.value.copy(
                                cameraOverlayInfo = info.first, inferenceTime = info.second
                            )
                        } else {
                            _uiState.value.copy(
                                galleryOverlayInfo = info.first, inferenceTime = info.second
                            )
                        }
                    }
                }
        }
        viewModelScope.launch {
            imageSegmentationHelper.error.collect { throwable ->
                _uiState.update {
                    _uiState.value.copy(errorMessage = throwable?.message)
                }
            }
        }
    }

    private var segmentJob: Job? = null

    private val _uiState: MutableStateFlow<UiState> = MutableStateFlow(UiState())

    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    /** Start segment an image.
     *  @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap,
     *  setting the new bitmap's config to Bitmap.Config.ARGB_8888
     */
    fun segment(bitmap: Bitmap) {
        segmentJob = viewModelScope.launch {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            imageSegmentationHelper.segment(argbBitmap)
        }
    }

    /** Stop current segmentation */
    private fun stopSegment() {
        viewModelScope.launch {
            segmentJob?.cancel()
            _uiState.update {
                if (it.currentTab == Tab.Camera) {
                    _uiState.value.copy(cameraOverlayInfo = null)
                } else {
                    _uiState.value.copy(galleryOverlayInfo = null)
                }
            }
        }
    }

    /** Update display gallery uri*/
    fun updateCurrentTab(tab: Tab) {
        _uiState.update { _uiState.value.copy(currentTab = tab) }
    }

    /** Update display gallery uri*/
    fun updateGalleryUri(uri: Uri) {
        if (uri != _uiState.value.galleryUri) {
            stopSegment()
        }
        _uiState.update { _uiState.value.copy(galleryUri = uri) }
    }

    /**
     * Save a new uri for each photo taken
     *  @param uri should be generated randomly to avoid skip recompose when taking a new image over the old one
     */
    fun updateCameraUriTemp(uri: Uri) {
        _uiState.update { _uiState.value.copy(cameraUriTemp = uri) }
    }

    /** Update display gallery uri*/
    fun updateCameraUri() {
        if (_uiState.value.cameraUriTemp != _uiState.value.cameraUri) {
            stopSegment()
        }
        _uiState.update {
            _uiState.value.copy(
                cameraUri = _uiState.value.cameraUriTemp, cameraOverlayInfo = null
            )
        }
    }


    /** Set Delegate for ImageSegmentationHelper(CPU/NNAPI)*/
    fun setDelegate(delegate: ImageSegmentationHelper.Delegate) {
        viewModelScope.launch {
            imageSegmentationHelper.initClassifier(delegate)
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        _uiState.update {
            _uiState.value.copy(errorMessage = null)
        }
    }
}