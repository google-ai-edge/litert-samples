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

package com.google.aiedge.examples.interactivesegmentation

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val helper: Sam2SegmentationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                return MainViewModel(Sam2SegmentationHelper(context)) as T
            }
        }
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    // Kept so we can re-encode the same image when the accelerator changes.
    private var currentBitmap: Bitmap? = null

    init {
        viewModelScope.launch { helper.initSegmenter() }
        viewModelScope.launch {
            helper.segmentation.collect { result ->
                _uiState.update {
                    it.copy(
                        maskBitmap = result.maskBitmap,
                        maskIou = result.iou,
                        decodeTimeMs = result.decodeTimeMs,
                    )
                }
            }
        }
        viewModelScope.launch {
            helper.error.collect { e -> _uiState.update { it.copy(errorMessage = e?.message) } }
        }
    }

    /** A new image was picked: show it and run the heavy encoder once. */
    fun onImagePicked(bitmap: Bitmap) {
        currentBitmap = bitmap
        _uiState.update {
            it.copy(imageBitmap = bitmap, maskBitmap = null, maskIou = 0f, isEncoding = true)
        }
        viewModelScope.launch {
            val time = helper.encodeImage(bitmap)
            _uiState.update { it.copy(encodeTimeMs = time, isEncoding = false) }
        }
    }

    /** The user tapped a point (in 1024x1024 model space): decode a mask there. */
    fun onTap(px: Float, py: Float) {
        val state = _uiState.value
        if (state.imageBitmap == null || state.isEncoding) return
        viewModelScope.launch { helper.segmentAt(px, py) }
    }

    /** Switch CPU/GPU: recreate the models and re-encode the current image. */
    fun setAccelerator(delegate: Sam2SegmentationHelper.AcceleratorEnum) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(setting = it.setting.copy(delegate = delegate), maskBitmap = null, maskIou = 0f)
            }
            helper.setOptions(Sam2SegmentationHelper.Options(delegate = delegate))
            helper.initSegmenter()
            currentBitmap?.let { bitmap ->
                _uiState.update { it.copy(isEncoding = true) }
                val time = helper.encodeImage(bitmap)
                _uiState.update { it.copy(encodeTimeMs = time, isEncoding = false) }
            }
        }
    }

    fun errorMessageShown() {
        _uiState.update { it.copy(errorMessage = null) }
    }
}
