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

package com.google.aiedge.examples.object_detection

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.aiedge.examples.object_detection.objectdetector.ObjectDetectorHelper
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val objectDetectorHelper: ObjectDetectorHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                // To apply object detection, we use our ObjectDetectorHelper class,
                // which abstracts away the specifics of using MediaPipe  for object
                // detection from the UI elements of the app
                val objectDetectorHelper = ObjectDetectorHelper(context = context)
                return MainViewModel(objectDetectorHelper) as T
            }
        }
    }

    private var detectJob: Job? = null

    private val detectionResult =
        MutableStateFlow<ObjectDetectorHelper.DetectionResult?>(null).also {
            viewModelScope.launch {
                objectDetectorHelper.detectionResult.collect(it)
            }
        }

    private val setting = MutableStateFlow(Setting())
        .apply {
            viewModelScope.launch {
                collect {
                    objectDetectorHelper.apply {
                        model = it.model
                        delegate = it.delegate
                        maxResults = it.resultCount
                        threshold = it.threshold
                    }
                    objectDetectorHelper.setupObjectDetector()
                }
            }
        }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            objectDetectorHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        detectionResult,
        setting,
        errorMessage
    ) { result, setting, error ->
        UiState(
            detectionResult = result,
            setting = setting,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    /**
     *  Start detect object from an image.
     *  @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap,
     *  @param rotationDegrees to correct the rotationDegrees during segmentation
     */
    fun detectImageObject(bitmap: Bitmap, rotationDegrees: Int) {
        detectJob = viewModelScope.launch {
            objectDetectorHelper.detect(bitmap, rotationDegrees)
        }
    }

    fun detectImageObject(imageProxy: ImageProxy) {
        detectJob = viewModelScope.launch {
            objectDetectorHelper.detect(imageProxy)
            imageProxy.close()
        }
    }

    /** Set [ObjectDetectorHelper.Delegate] (CPU/GPU) for ObjectDetectorHelper*/
    fun setDelegate(delegate: ObjectDetectorHelper.Delegate) {
        viewModelScope.launch {
            setting.update { it.copy(delegate = delegate) }
        }
    }

    /** Set Number of output classes of the ObjectDetectorHelper.  */
    fun setNumberOfResult(numResult: Int) {
        viewModelScope.launch {
            setting.update { it.copy(resultCount = numResult) }
        }
    }

    /** Set the threshold so the label can display score */
    fun setThreshold(threshold: Float) {
        viewModelScope.launch {
            setting.update { it.copy(threshold = threshold) }
        }
    }

    /** Stop current detection */
    fun stopDetect() {
        viewModelScope.launch {
            detectionResult.emit(null)
            detectJob?.cancel()
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}