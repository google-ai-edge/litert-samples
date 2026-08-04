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

package com.google.ai.edge.examples.text_to_speech_streaming

import androidx.compose.runtime.Immutable

/**
 * Live synthesis numbers for the metrics card. TTFA (time-to-first-audio) is the headline: tap ->
 * first PCM handed to the AudioTrack. RTF (real-time factor) is total synthesis compute over total
 * audio duration — below 1.0 means the stream keeps ahead of playback.
 */
@Immutable
data class Metrics(
  val ttfaMs: Long = 0,
  val rtf: Double = 0.0,
  val audioSeconds: Double = 0.0,
  val synthMs: Long = 0,
  val sentencesDone: Int = 0,
  val sentencesTotal: Int = 0,
)

/**
 * Immutable snapshot of everything the screen renders. Audio is played back as a side effect from
 * the ViewModel and is not part of this state.
 */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isSpeaking: Boolean = false,
  val statusMessage: String = "",
  val errorMessage: String? = null,
  val voices: List<String> = emptyList(),
  val selectedVoice: Int = 0,
  val modelMegabytes: Double = 0.0,
  val metrics: Metrics? = null,
)
