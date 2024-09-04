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

import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
    val classifications: List<Category> = listOf(),
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val inferenceTime: Long = 0L,
    val model: AudioClassificationHelper.TFLiteModel = AudioClassificationHelper.DEFAULT_MODEL,
    val delegate: AudioClassificationHelper.Delegate = AudioClassificationHelper.DEFAULT_DELEGATE,
    val overlap: Float = AudioClassificationHelper.DEFAULT_OVERLAP,
    val resultCount: Int = AudioClassificationHelper.DEFAULT_RESULT_COUNT,
    val threshold: Float = AudioClassificationHelper.DEFAULT_PROBABILITY_THRESHOLD,
    val threadCount: Int = AudioClassificationHelper.DEFAULT_THREAD_COUNT
)
