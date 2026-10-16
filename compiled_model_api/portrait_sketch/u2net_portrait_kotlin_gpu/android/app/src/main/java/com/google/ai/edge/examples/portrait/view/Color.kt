/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.portrait.view

import androidx.compose.ui.graphics.Color

val darkBlue = Color(0xFF020F59)
val teal = Color(0xFF00C99E)

/** Distinct colors cycled per object class when drawing detection boxes. */
val boxPalette =
  intArrayOf(
    0xFF00C853.toInt(),
    0xFFFF6D00.toInt(),
    0xFF2962FF.toInt(),
    0xFFD50000.toInt(),
    0xFFAA00FF.toInt(),
    0xFF00B8D4.toInt(),
    0xFFFFD600.toInt(),
    0xFFC51162.toInt(),
  )
