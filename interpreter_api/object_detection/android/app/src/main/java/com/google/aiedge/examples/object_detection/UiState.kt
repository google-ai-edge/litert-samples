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

package com.google.aiedge.examples.object_detection

import androidx.compose.runtime.Immutable
import com.google.aiedge.examples.object_detection.objectdetector.ObjectDetectorHelper

@Immutable
class UiState(
    val detectionResult: ObjectDetectorHelper.DetectionResult? = null,
    val setting: Setting = Setting(),
    val errorMessage: String? = null,
)

@Immutable
data class Setting(
    val model: ObjectDetectorHelper.Model = ObjectDetectorHelper.MODEL_DEFAULT,
    val delegate: ObjectDetectorHelper.Delegate = ObjectDetectorHelper.Delegate.CPU,
    val resultCount: Int = ObjectDetectorHelper.MAX_RESULTS_DEFAULT,
    val threshold: Float = ObjectDetectorHelper.THRESHOLD_DEFAULT,
)
