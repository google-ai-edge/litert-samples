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

package com.google.ai.edge.examples.low_light_enhancement.view

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Small "Input" / "Enhanced" caption pinned to the top-left of a pane. */
@Composable
fun BoxScope.PaneLabel(text: String) {
  Text(
    text = text,
    color = Color.White,
    fontSize = 12.sp,
    modifier =
      Modifier.align(Alignment.TopStart)
        .padding(8.dp)
        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
        .padding(horizontal = 8.dp, vertical = 4.dp),
  )
}
