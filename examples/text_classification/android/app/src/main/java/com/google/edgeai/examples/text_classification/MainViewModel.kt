/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

package com.google.edgeai.examples.text_classification

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

class MainViewModel(private val textClassificationHelper: TextClassificationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val textClassificationHelper = TextClassificationHelper(context)
                return MainViewModel(textClassificationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch {
            textClassificationHelper.initClassifier()
        }
    }

    private var classifyJob: Job? = null

    private val percentages = textClassificationHelper.percentages.distinctUntilChanged().stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), Pair(FloatArray(0), 0L)
    )

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            textClassificationHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        percentages, errorMessage
    ) { percentages, throwable ->
        textClassificationHelper.completableDeferred?.complete(Unit)
        val percents = percentages.first
        UiState(
            negativePercentage = if (percents.isNotEmpty()) percents.first() else 0f,
            positivePercentage = if (percents.isNotEmpty()) percents.last() else 0f,
            inferenceTime = percentages.second,
            errorMessage = throwable?.message
        )
    }.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
    )

    /**
     * Cancel current job and create a new coroutine to classify the new input text
     */
    fun runClassification(inputText: String) {
        classifyJob?.cancel()
        classifyJob = viewModelScope.launch {
            textClassificationHelper.classify(inputText)
        }
    }

    /*
     * Stop classification and setup Interpreter with new model
     */
    fun setModel(model: TextClassificationHelper.TFLiteModel) {
        viewModelScope.launch {
            textClassificationHelper.stopClassify()
            textClassificationHelper.initClassifier(model)
        }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}