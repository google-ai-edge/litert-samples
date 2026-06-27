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

package com.google.aiedge.examples.superresolution

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val helper: SuperResolutionHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                return MainViewModel(SuperResolutionHelper(context)) as T
            }
        }
    }

    private val setting = MutableStateFlow(Setting()).apply {
        viewModelScope.launch {
            collect {
                helper.setOptions(SuperResolutionHelper.Options(model = it.model, delegate = it.delegate))
                helper.initModel()
            }
        }
    }

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch { helper.error.collect(it) }
    }

    private val resultFlow = MutableStateFlow<SuperResolutionHelper.Result?>(null).also {
        viewModelScope.launch { helper.result.collect(it) }
    }

    val uiState: StateFlow<UiState> = combine(
        resultFlow,
        setting.filterNotNull(),
        errorMessage,
    ) { result, setting, error ->
        UiState(
            superResolved = result?.superResolved,
            baseline = result?.baseline,
            inferenceTime = result?.inferenceTime ?: 0L,
            setting = setting,
            errorMessage = error?.message,
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    fun superResolve(bitmap: Bitmap) {
        viewModelScope.launch {
            helper.superResolve(bitmap.copy(Bitmap.Config.ARGB_8888, false))
        }
    }

    fun setAccelerator(delegate: SuperResolutionHelper.AcceleratorEnum) {
        viewModelScope.launch { setting.update { it.copy(delegate = delegate) } }
    }

    fun setModel(model: SuperResolutionHelper.Model) {
        viewModelScope.launch { setting.update { it.copy(model = model) } }
    }

    fun errorMessageShown() { errorMessage.update { null } }
}
