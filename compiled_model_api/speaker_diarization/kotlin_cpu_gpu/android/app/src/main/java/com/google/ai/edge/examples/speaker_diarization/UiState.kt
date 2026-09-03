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

package com.google.ai.edge.examples.speaker_diarization

import android.graphics.Bitmap
import androidx.compose.runtime.Immutable

/** One speaker's total talk time and the ARGB color used for its timeline row and play button. */
@Immutable data class SpeakerRow(val speaker: Int, val seconds: Double, val color: Int)

/** Immutable snapshot of everything the diarization screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isRecording: Boolean = false,
  val isAnalyzing: Boolean = false,
  /** The colored per-speaker timeline, drawn host-side; null until a clip is diarized. */
  val timeline: Bitmap? = null,
  val speakers: List<SpeakerRow> = emptyList(),
  val statusMessage: String = "",
  val errorMessage: String? = null,
)
