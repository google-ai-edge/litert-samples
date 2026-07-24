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

package com.google.aiedge.examples.imageclassification

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

class MainViewModel(private val imageClassificationHelper: ImageClassificationHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageClassificationHelper = ImageClassificationHelper(context)
                return MainViewModel(imageClassificationHelper) as T
            }
        }
    }

    private var classificationJob: Job? = null

    private val setting = MutableStateFlow(Setting())
        .apply {
            viewModelScope.launch {
                collect {
                    imageClassificationHelper.setOptions(
                        ImageClassificationHelper.Options(
                            model = it.model,
                            delegate = it.delegate,
                            resultCount = it.resultCount,
                            probabilityThreshold = it.threshold
                        )
                    )
                    imageClassificationHelper.initClassifier()
                }
            }
        }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            imageClassificationHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        imageClassificationHelper.classification
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5_000),
                ImageClassificationHelper.ClassificationResult(emptyList(), 0L)
            ),
        setting.filterNotNull(),
        errorMessage,
    ) { result, setting, error ->
        UiState(
            inferenceTime = result.inferenceTime,
            categories = result.categories,
            setting = setting,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())


    /** Start classify an image.
     *  @param imageProxy contain `imageBitMap` and imageInfo as `image rotation degrees`.
     *
     */
    fun classify(imageProxy: ImageProxy) {
        classificationJob = viewModelScope.launch {
            imageClassificationHelper.classify(
                imageProxy.toBitmap(),
                imageProxy.imageInfo.rotationDegrees,
            )
            imageProxy.close()
        }
    }

    /** Start classify an image.
     *  @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap,
     *  setting the new bitmap's config to Bitmap.Config.ARGB_8888
     *  @param rotationDegrees to correct the rotationDegrees during classification
     */
    fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        viewModelScope.launch {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            imageClassificationHelper.classify(argbBitmap, rotationDegrees)
        }
    }

    /** Stop current classification */
    fun stopClassify() {
        classificationJob?.cancel()
    }

    /** Set [ImageClassificationHelper.Delegate] (CPU/NNAPI) for ImageSegmentationHelper*/
    fun setDelegate(delegate: ImageClassificationHelper.Delegate) {
        viewModelScope.launch {
            setting.update { it.copy(delegate = delegate) }
        }
    }

    /** Set [ImageClassificationHelper.Model] for ImageSegmentationHelper*/
    fun setModel(model: ImageClassificationHelper.Model) {
        viewModelScope.launch {
            setting.update { it.copy(model = model) }
        }
    }

    /** Set Number of output classes of the [ImageClassificationHelper.Model].  */
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

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}