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

package com.google.aiedge.examples.featurematching

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val helper: XFeatHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                return MainViewModel(XFeatHelper(context)) as T
            }
        }
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var imageA: Bitmap? = null
    private var imageB: Bitmap? = null

    init {
        viewModelScope.launch { helper.init() }
    }

    fun setImage(slotA: Boolean, bitmap: Bitmap) {
        val bmp = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        if (slotA) imageA = bmp else imageB = bmp
        _uiState.update { it.copy(imageA = imageA, imageB = imageB) }
        runMatch()
    }

    fun setDelegate(d: XFeatHelper.AcceleratorEnum) {
        helper.setDelegate(d)
        _uiState.update { it.copy(delegate = d) }
        viewModelScope.launch { helper.init(); runMatch() }
    }

    private fun runMatch() {
        val a = imageA; val b = imageB
        if (a == null || b == null) return
        viewModelScope.launch {
            try {
                val t0 = SystemClock.uptimeMillis()
                val fa = helper.extract(a)
                val fb = helper.extract(b)
                val pairs = helper.match(fa, fb)
                val dt = SystemClock.uptimeMillis() - t0
                val matches = pairs.map { (i, k) ->
                    Match(fa[i].x, fa[i].y, fb[k].x, fb[k].y)
                }
                _uiState.update { it.copy(matches = matches, inferenceTime = dt) }
            } catch (e: Exception) {
                _uiState.update { it.copy(errorMessage = e.message) }
            }
        }
    }

    fun errorMessageShown() { _uiState.update { it.copy(errorMessage = null) } }
}
