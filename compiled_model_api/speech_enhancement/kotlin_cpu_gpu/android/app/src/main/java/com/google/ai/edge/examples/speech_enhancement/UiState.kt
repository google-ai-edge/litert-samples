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

package com.google.ai.edge.examples.speech_enhancement

import androidx.compose.runtime.Immutable

/**
 * Immutable snapshot of everything the speech-enhancement screen renders. [noisy] holds the source
 * clip (mic or picked file) and [clean] holds the CMGAN-enhanced result; both are played back as a
 * side effect from the ViewModel through an [android.media.AudioTrack], so the arrays are here only
 * to enable the two Play buttons once their audio exists.
 */
@Immutable
data class UiState(
  val isModelReady: Boolean = false,
  val isRecording: Boolean = false,
  val isEnhancing: Boolean = false,
  val statusMessage: String = "",
  val errorMessage: String? = null,
  val noisy: FloatArray? = null,
  val clean: FloatArray? = null,
)
