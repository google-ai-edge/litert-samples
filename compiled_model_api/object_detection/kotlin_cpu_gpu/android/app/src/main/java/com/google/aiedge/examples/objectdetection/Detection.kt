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

package com.google.aiedge.examples.objectdetection

/** A single detection. Box coordinates are in upright-image pixels (see DetectionResult). */
data class Detection(
    val classId: Int,
    val label: String,
    val score: Float,
    val xMin: Float,
    val yMin: Float,
    val xMax: Float,
    val yMax: Float,
)
