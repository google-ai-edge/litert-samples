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

package com.google.aiedge.examples.zeroshotimageclassification

import androidx.compose.runtime.Immutable

@Immutable
class UiState(
    val inferenceTime: Long = 0L,
    val categories: List<ZeroShotImageClassificationHelper.Category> = emptyList(),
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val model: ZeroShotImageClassificationHelper.Model = ZeroShotImageClassificationHelper.DEFAULT_MODEL,
    val delegate: ZeroShotImageClassificationHelper.AcceleratorEnum = ZeroShotImageClassificationHelper.DEFAULT_DELEGATE,
    val resultCount: Int = ZeroShotImageClassificationHelper.DEFAULT_RESULT_COUNT,
    val threshold: Float = ZeroShotImageClassificationHelper.DEFAULT_THRESHOLD,
    val threadCount: Int = ZeroShotImageClassificationHelper.DEFAULT_THREAD_COUNT
)