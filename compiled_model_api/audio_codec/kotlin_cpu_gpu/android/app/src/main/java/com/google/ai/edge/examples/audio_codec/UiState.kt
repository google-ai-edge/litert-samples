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

package com.google.ai.edge.examples.audio_codec

import androidx.compose.runtime.Immutable

/**
 * Immutable snapshot of everything the codec screen renders. Both waveforms are kept so the two
 * Play buttons can replay them. The audio is played back as a side effect from the ViewModel.
 *
 * @property isModelReady whether the round-trip finished and the Play buttons should be enabled.
 * @property original the decoded bundled input clip (mono float32, 24 kHz).
 * @property reconstructed the codec output for the same clip (mono float32, 24 kHz).
 * @property statusMessage the human-readable status line (timings and RTF).
 * @property errorMessage a load/inference failure to surface instead of the status, if any.
 */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val original: FloatArray = FloatArray(0),
  val reconstructed: FloatArray = FloatArray(0),
  val statusMessage: String = "",
  val errorMessage: String? = null,
)
