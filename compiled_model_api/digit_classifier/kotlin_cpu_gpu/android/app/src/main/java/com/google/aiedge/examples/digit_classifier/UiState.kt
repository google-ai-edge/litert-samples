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

import androidx.compose.runtime.Immutable
import com.google.aiedge.examples.digit_classifier.DigitClassificationHelper.AcceleratorEnum

@Immutable
class UiState(
    val digit: String = "-",
    val score: Float = 0f,
    val drawOffsets: List<DrawOffset> = emptyList(),
    val accelerator: AcceleratorEnum = AcceleratorEnum.CPU
)

abstract class DrawOffset

data class Start(val x: Float, val y: Float) : DrawOffset()
data class Point(val x: Float, val y: Float) : DrawOffset()