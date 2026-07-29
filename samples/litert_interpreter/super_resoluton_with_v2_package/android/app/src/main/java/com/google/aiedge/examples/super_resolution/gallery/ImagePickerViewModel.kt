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

package com.google.aiedge.examples.super_resolution.gallery

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.aiedge.examples.super_resolution.ImageSuperResolutionHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

class ImagePickerViewModel(private val imageSuperResolutionHelper: ImageSuperResolutionHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageSuperResolutionHelper = ImageSuperResolutionHelper(context)
                return ImagePickerViewModel(imageSuperResolutionHelper) as T
            }
        }
    }

    private val originalBimapFlow = MutableStateFlow<Bitmap?>(null)
    private val selectedPointFlow = MutableStateFlow(SelectPoint())
    private val superResolutionFlow = MutableStateFlow(ImageSuperResolutionHelper.Result())

    init {
        viewModelScope.launch {
            imageSuperResolutionHelper.superResolutionFlow.collect { result ->
                superResolutionFlow.update {
                    it.bitmap?.recycle()
                    result
                }
            }
        }
    }

    private var sharpenJob: Job? = null

    val uiState: StateFlow<ImagePickerUiState> = combine(
        originalBimapFlow,
        selectedPointFlow,
        superResolutionFlow,
    ) { originalBitmap, selectedPoint, result ->
        ImagePickerUiState(
            originalBitmap = originalBitmap,
            sharpenBitmap = result.bitmap,
            selectPoint = selectedPoint,
            inferenceTime = result.inferenceTime.toInt()
        )
    }.stateIn(viewModelScope, SharingStarted.Lazily, ImagePickerUiState())

    /**
     * Processes the selected image and emits the original bitmap
     */
    fun selectImage(bitmap: Bitmap) {
        viewModelScope.launch {
            clear()
            originalBimapFlow.emit(bitmap)
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    fun selectOffset(offset: Offset, imageSize: Size) {
        if (sharpenJob?.isCompleted == false) return
        sharpenJob = viewModelScope.launch(Dispatchers.IO) {
            flowOf(offset).onEach {
                val ratioW = originalBimapFlow.value!!.width.toFloat() / imageSize.width
                val x = offset.x - ImagePickerUiState.REQUIRE_IMAGE_SIZE / 2 / ratioW
                val y = offset.y - ImagePickerUiState.REQUIRE_IMAGE_SIZE / 2 / ratioW
                selectedPointFlow.emit(
                    SelectPoint(
                        Offset(x, y), ImagePickerUiState.REQUIRE_IMAGE_SIZE / ratioW
                    )
                )
            }.flatMapLatest {
                flowOf(
                    getSubBitmap(
                        originalBitmap = originalBimapFlow.value!!,
                        offset = it,
                        imageSize = imageSize,
                    )
                )
            }.collectLatest(::makeSharpen)
        }
    }

    /** Set [ImageSuperResolutionHelper.Delegate] (CPU/NNAPI) for ImageSuperResolutionHelper*/
    fun setDelegate(delegate: ImageSuperResolutionHelper.Delegate) {
        imageSuperResolutionHelper.initClassifier(delegate)
    }

    /**
     * Starts the process of sharpening the provided bitmap.
     */
    private suspend fun makeSharpen(bitmap: Bitmap) {
        if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.setConfig(Bitmap.Config.ARGB_8888)
        }
        imageSuperResolutionHelper.makeSuperResolution(bitmap)
    }

    private fun getSubBitmap(originalBitmap: Bitmap, offset: Offset, imageSize: Size): Bitmap {
        val bitmapOffset = convertToBitmapOffset(offset, imageSize)
        val requireSize = ImagePickerUiState.REQUIRE_IMAGE_SIZE

        // Ensure the x and y coordinates are within bounds
        val x = (bitmapOffset.x - requireSize / 2).coerceIn(
            0f, (originalBitmap.width - requireSize / 2)
        ).roundToInt()
        val y = (bitmapOffset.y - requireSize / 2).coerceIn(
            0f, (originalBitmap.height - requireSize / 2)
        ).roundToInt()
        return Bitmap.createBitmap(
            originalBitmap, x, y, requireSize.toInt(), requireSize.toInt()
        )
    }

    // Clear selected point & sharpen bitmap
    private fun clear() {
        viewModelScope.launch {
            selectedPointFlow.emit(SelectPoint())
            superResolutionFlow.emit(ImageSuperResolutionHelper.Result())
        }
    }

    private fun convertToBitmapOffset(
        tapOffset: Offset, imageSize: Size
    ): Offset {
        val ratio = originalBimapFlow.value!!.width.toFloat() / imageSize.width
        val x = tapOffset.x * ratio
        val y = tapOffset.y * ratio

        return Offset(x, y)
    }
}
