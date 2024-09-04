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

package com.google.aiedge.examples.audio_classification

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
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val audioClassificationHelper: AudioClassificationHelper) :
    ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val audioClassificationHelper = AudioClassificationHelper(context)
                return MainViewModel(audioClassificationHelper) as T
            }
        }
    }

    private var job: Job? = null

    private val errorMessage = MutableStateFlow<Throwable?>(null)

    private val setting = MutableStateFlow<Setting?>(null).apply {
        viewModelScope.launch {
            collect {
                if (it == null) {
                    return@collect
                }
                try {
                    stopClassifier()
                    audioClassificationHelper.options.currentModel = it.model
                    audioClassificationHelper.options.delegate = it.delegate
                    audioClassificationHelper.options.overlapFactor = it.overlap
                    audioClassificationHelper.options.resultCount = it.resultCount
                    audioClassificationHelper.options.probabilityThreshold = it.threshold
                    audioClassificationHelper.options.threadCount = it.threadCount
                    audioClassificationHelper.setupInterpreter()
                    job = launch {
                        audioClassificationHelper.startRecord()
                    }
                } catch (e: Exception) {
                    errorMessage.emit(e)
                }
            }
        }
    }
    private val probabilities =
        audioClassificationHelper.probabilities.stateIn(
            viewModelScope, SharingStarted.WhileSubscribed(5_000), Pair(
                listOf(), 0L
            )
        )

    val uiState: StateFlow<UiState> =
        combine(
            setting.filterNotNull(),
            probabilities,
            errorMessage
        ) { setting, probabilities, throwable ->
            UiState(
                classifications = probabilities.first,
                setting = setting.copy(inferenceTime = probabilities.second),
                errorMessage = throwable?.message
            )
        }.stateIn(
            viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
        )

    fun startAudioClassification() {
        setting.update { Setting() }
    }

    fun setModel(model: AudioClassificationHelper.TFLiteModel) {
        setting.update { it?.copy(model = model) }
    }

    fun setDelegate(delegate: AudioClassificationHelper.Delegate) {
        setting.update { it?.copy(delegate = delegate) }
    }

    fun setOverlap(value: Float) {
        setting.update { it?.copy(overlap = value) }
    }

    fun setMaxResults(max: Int) {
        setting.update { it?.copy(resultCount = max) }
    }

    fun setThreshold(threshold: Float) {
        setting.update { it?.copy(threshold = threshold) }
    }

    fun setThreadCount(count: Int) {
        setting.update { it?.copy(threadCount = count) }
    }

    fun throwError(error: Throwable) {
        errorMessage.update { error }
    }

    /** Clear error message after it has been consumed*/
    fun errorMessageShown() {
        errorMessage.update { null }
    }

    fun stopClassifier() {
        audioClassificationHelper.stop()
        if (job?.isActive == true) {
            job?.cancel()
        }
    }
}
