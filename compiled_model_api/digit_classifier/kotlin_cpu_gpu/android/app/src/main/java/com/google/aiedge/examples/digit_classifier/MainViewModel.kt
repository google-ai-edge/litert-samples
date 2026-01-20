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

package com.google.aiedge.examples.digit_classifier

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
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val digitClassificationHelper: DigitClassificationHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val digitClassificationHelper = DigitClassificationHelper(context)
                return MainViewModel(digitClassificationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch {
            digitClassificationHelper.initClassifier()
        }
    }

    private val drawFlow = MutableStateFlow<List<DrawOffset>>(emptyList())
    private val acceleratorFlow = MutableStateFlow(DigitClassificationHelper.AcceleratorEnum.CPU)

    val uiState: StateFlow<UiState> = combine(
        digitClassificationHelper.classification
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5_000),
                Pair("-", 0f)
            ),
        drawFlow,
        acceleratorFlow
    ) { pair, drawOffsets, accelerator ->
        UiState(
            digit = if (drawOffsets.isEmpty()) "-" else pair.first,
            score = if (drawOffsets.isEmpty()) 0f else pair.second,
            drawOffsets = drawOffsets,
            accelerator = accelerator
        )
    }.stateIn(viewModelScope, SharingStarted.Lazily, UiState())

    fun classify(bitmap: Bitmap) {
        viewModelScope.launch {
            digitClassificationHelper.classify(bitmap)
        }
    }

    fun draw(drawOffset: DrawOffset) {
        drawFlow.update {
            it + drawOffset
        }
    }

    /* Clean the board*/
    fun cleanBoard() {
        drawFlow.update { emptyList() }
    }

    /** Set Accelerator for DigitClassificationHelper(CPU/GPU) */
    fun setAccelerator(acceleratorEnum: DigitClassificationHelper.AcceleratorEnum) {
        viewModelScope.launch { 
            digitClassificationHelper.initClassifier(acceleratorEnum)
            acceleratorFlow.update { acceleratorEnum }
        }
    }
}
