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

package com.google.aiedge.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
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
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val objectDetectorHelper: ObjectDetectorHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val objectDetectorHelper = ObjectDetectorHelper(context)
                return MainViewModel(objectDetectorHelper) as T
            }
        }
    }

    private var detectionJob: Job? = null

    private val setting = MutableStateFlow(Setting())
        .apply {
            viewModelScope.launch {
                collect {
                    objectDetectorHelper.setOptions(
                        ObjectDetectorHelper.Options(
                            model = it.model,
                            delegate = it.delegate,
                            threshold = it.threshold,
                        )
                    )
                    objectDetectorHelper.initDetector()
                }
            }
        }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            objectDetectorHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        objectDetectorHelper.detection
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5_000),
                ObjectDetectorHelper.DetectionResult(emptyList(), 0L, 0, 0)
            ),
        setting.filterNotNull(),
        errorMessage,
    ) { result, setting, error ->
        UiState(
            inferenceTime = result.inferenceTime,
            detections = result.detections,
            imageWidth = result.imageWidth,
            imageHeight = result.imageHeight,
            setting = setting,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    /** Detect objects in a camera frame. */
    fun detect(imageProxy: ImageProxy) {
        detectionJob = viewModelScope.launch {
            objectDetectorHelper.detect(
                imageProxy.toBitmap(),
                imageProxy.imageInfo.rotationDegrees,
            )
            imageProxy.close()
        }
    }

    /** Detect objects in a still bitmap (gallery). */
    fun detect(bitmap: Bitmap, rotationDegrees: Int) {
        viewModelScope.launch {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            objectDetectorHelper.detect(argbBitmap, rotationDegrees)
        }
    }

    /** Stop the current detection. */
    fun stopDetect() {
        detectionJob?.cancel()
    }

    fun setAccelerator(delegate: ObjectDetectorHelper.AcceleratorEnum) {
        viewModelScope.launch {
            setting.update { it.copy(delegate = delegate) }
        }
    }

    fun setModel(model: ObjectDetectorHelper.Model) {
        viewModelScope.launch {
            setting.update { it.copy(model = model) }
        }
    }

    fun setThreshold(threshold: Float) {
        viewModelScope.launch {
            setting.update { it.copy(threshold = threshold) }
        }
    }

    /** Clear the error message after it has been consumed. */
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}
