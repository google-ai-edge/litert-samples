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

package com.google.ai.edge.examples.zero_shot_classification

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Bundled upstream question sets; their schemas remain in presets.json. */
enum class Preset(val assetKey: String, @param:StringRes val title: Int) {
  EMAIL("email", R.string.preset_email),
  TRIAGE("triage", R.string.preset_triage),
  MODERATION("moderation", R.string.preset_moderation),
}

/** Language of the invented example currently loaded in the state editor. */
enum class ExampleLanguage(@param:StringRes val title: Int) {
  JA(R.string.language_japanese),
  EN(R.string.language_english),
}

/** One decoded option, with the original label or a localized boolean/score label. */
@Immutable
data class ProbabilityUiRow(
  val probability: Double,
  val label: String = "",
  @param:StringRes val labelResource: Int? = null,
  val scoreLevel: Int? = null,
  val description: String? = null,
)

/** Display data for one question; numerical values come from the host decoder. */
@Immutable
data class AnswerUiRow(
  val questionId: String,
  val instructions: String,
  val type: String,
  val choice: String? = null,
  val score: Double? = null,
  val trueProbability: Double? = null,
  val probabilities: List<ProbabilityUiRow>,
  val confidence: Double,
  val totalMs: Double,
  val tokenCount: Int,
)

/** Immutable UI snapshot emitted by the model-owning ViewModel. */
@Immutable
data class UiState(
  val inputSubject: String,
  val inputText: String,
  val preset: Preset = Preset.EMAIL,
  val language: ExampleLanguage = ExampleLanguage.EN,
  val accelerator: LayaEngine.Backend = LayaEngine.Backend.GPU,
  /** LiteRT lists this SoC and the NPU runtime module is installed; otherwise NPU is disabled. */
  val npuAvailable: Boolean = false,
  val calibrated: Boolean = true,
  val busy: Boolean = true,
  val ready: Boolean = false,
  /** The model files are missing, so the header shows the download controls. */
  val downloadNeeded: Boolean = false,
  val downloading: Boolean = false,
  /** A download attempt failed. The button reads Retry and resumes from the .part files. */
  val downloadFailed: Boolean = false,
  val downloadedBytes: Long = 0L,
  val downloadTotalBytes: Long = 0L,
  val meteredNetwork: Boolean = false,
  /** Set when the free space check failed. The header then shows both numbers and Retry. */
  val spaceNeededBytes: Long? = null,
  val spaceAvailableBytes: Long? = null,
  @param:StringRes val statusMessage: Int = R.string.status_loading_tokenizer,
  val answers: List<AnswerUiRow> = emptyList(),
  val runTotalMs: Double? = null,
  val runTokenCount: Int = 0,
  val launchToReadyMs: Double? = null,
  val errorMessage: String? = null,
  /** The accelerator offered after a compile failure: GPU after the NPU, CPU after the GPU. */
  val fallback: LayaEngine.Backend? = null,
)
