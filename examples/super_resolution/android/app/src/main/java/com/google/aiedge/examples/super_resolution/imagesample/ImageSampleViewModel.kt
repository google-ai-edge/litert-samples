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

package com.google.aiedge.examples.super_resolution.imagesample

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.aiedge.examples.super_resolution.ImageSuperResolutionHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class ImageSampleViewModel(private val imageSuperResolutionHelper: ImageSuperResolutionHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSuperResolutionHelper = ImageSuperResolutionHelper(context)
                return ImageSampleViewModel(imageSuperResolutionHelper) as T
            }
        }
    }

    private val selectBitmapFlow = MutableStateFlow<Bitmap?>(null)
    private val superResolutionFlow = MutableStateFlow(ImageSuperResolutionHelper.Result())
    private var sharpenJob: Job? = null

    init {
        viewModelScope.launch {
            imageSuperResolutionHelper.superResolutionFlow
                .collect(superResolutionFlow)
        }
    }

    val uiState: StateFlow<ImageSampleUiState> = combine(
        selectBitmapFlow,
        superResolutionFlow,
    ) { bitmap, result ->
        ImageSampleUiState(
            selectBitmap = bitmap,
            sharpenBitmap = result.bitmap,
            inferenceTime = result.inferenceTime.toInt()
        )
    }.stateIn(viewModelScope, SharingStarted.Lazily, ImageSampleUiState())

    /*
     * Updates the selected image and resets the super-resolution process.
     */
    fun selectImage(bitmap: Bitmap) {
        superResolutionFlow.update { ImageSuperResolutionHelper.Result() }
        selectBitmapFlow.update {
            it?.recycle()
            bitmap
        }
    }

    /**
     * Starts the process of sharpening the provided bitmap.
     */
    fun makeSharpen() {
        if (sharpenJob?.isCompleted == false) return
        val bitmap = selectBitmapFlow.value ?: return
        sharpenJob = viewModelScope.launch(Dispatchers.IO) {
            if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.config = Bitmap.Config.ARGB_8888
            }
            imageSuperResolutionHelper.makeSuperResolution(bitmap)
        }
    }

    /** Set [ImageSuperResolutionHelper.Delegate] (CPU/NNAPI) for ImageSuperResolutionHelper*/
    fun setDelegate(delegate: ImageSuperResolutionHelper.Delegate) {
        imageSuperResolutionHelper.initClassifier(delegate)
    }
}
