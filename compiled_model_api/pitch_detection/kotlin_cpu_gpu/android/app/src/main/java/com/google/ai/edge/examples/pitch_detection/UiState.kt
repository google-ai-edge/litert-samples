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

package com.google.ai.edge.examples.pitch_detection

import androidx.compose.runtime.Immutable

/** Immutable snapshot of everything the tuner screen renders. */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isListening: Boolean = false,
  /** True once the mic loop has a confident pitch to show; false during silence. */
  val hasPitch: Boolean = false,
  /** True when the current note is within ±5 cents (drives the green in-tune coloring). */
  val isInTune: Boolean = false,
  /** e.g. "A4"; empty until [hasPitch]. */
  val note: String = "",
  /** A ±50-cent gauge drawn with monospace glyphs, or empty during silence. */
  val centsGauge: String = "",
  /** e.g. "440.0 Hz   +3 cents", or a "listening…" hint during silence. */
  val hzText: String = "",
  val statusMessage: String = "",
  val errorMessage: String? = null,
)
