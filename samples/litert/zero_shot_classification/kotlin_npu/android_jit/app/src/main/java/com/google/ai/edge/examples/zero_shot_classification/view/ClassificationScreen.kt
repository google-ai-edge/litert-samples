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

package com.google.ai.edge.examples.zero_shot_classification.view

import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.Divider
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Switch
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.google.ai.edge.examples.zero_shot_classification.AnswerUiRow
import com.google.ai.edge.examples.zero_shot_classification.ExampleLanguage
import com.google.ai.edge.examples.zero_shot_classification.LayaEngine
import com.google.ai.edge.examples.zero_shot_classification.Preset
import com.google.ai.edge.examples.zero_shot_classification.ProbabilityUiRow
import com.google.ai.edge.examples.zero_shot_classification.R
import com.google.ai.edge.examples.zero_shot_classification.UiState

/**
 * Product screen for local multilingual classification; all model work belongs to the ViewModel.
 */
@Composable
fun ClassificationScreen(
  state: UiState,
  onSubjectChange: (String) -> Unit,
  onInputChange: (String) -> Unit,
  onLanguage: (ExampleLanguage) -> Unit,
  onPreset: (Preset) -> Unit,
  onAccelerator: (LayaEngine.Backend) -> Unit,
  onCalibration: (Boolean) -> Unit,
  onRun: () -> Unit,
  onDownload: () -> Unit,
) {
  val editable = !state.busy && !state.downloadNeeded
  Scaffold(
    modifier = Modifier.statusBarsPadding().navigationBarsPadding().imePadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { insets ->
    Column(modifier = Modifier.fillMaxSize().padding(insets)) {
      StatusHeader(state, onAccelerator, onDownload)
      LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
      ) {
        item { PresetPicker(state.preset, editable, onPreset) }
        item {
          Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
              modifier = Modifier.fillMaxWidth(),
              horizontalArrangement = Arrangement.spacedBy(8.dp),
              verticalAlignment = Alignment.CenterVertically,
            ) {
              Text(
                stringResource(R.string.example_language),
                style = MaterialTheme.typography.subtitle2,
              )
              ExampleLanguage.entries.forEach { language ->
                if (state.language == language) {
                  Button(onClick = { onLanguage(language) }, enabled = editable) {
                    Text(stringResource(language.title))
                  }
                } else {
                  OutlinedButton(onClick = { onLanguage(language) }, enabled = editable) {
                    Text(stringResource(language.title))
                  }
                }
              }
            }
            if (state.preset == Preset.EMAIL) {
              OutlinedTextField(
                value = state.inputSubject,
                onValueChange = onSubjectChange,
                label = { Text(stringResource(R.string.email_subject)) },
                modifier = Modifier.fillMaxWidth(),
                enabled = editable,
                singleLine = true,
              )
            }
            OutlinedTextField(
              value = state.inputText,
              onValueChange = onInputChange,
              label = {
                Text(
                  stringResource(
                    when (state.preset) {
                      Preset.EMAIL -> R.string.email_body
                      Preset.TRIAGE -> R.string.support_message
                      Preset.MODERATION -> R.string.moderation_post
                    }
                  )
                )
              },
              modifier = Modifier.fillMaxWidth(),
              enabled = editable,
              minLines = 4,
              maxLines = 8,
            )
          }
        }
        item { AcceleratorPicker(state.accelerator, state.npuAvailable, editable, onAccelerator) }
        item {
          Row(
            modifier =
              Modifier.fillMaxWidth()
                .toggleable(
                  value = state.calibrated,
                  enabled = editable,
                  role = Role.Switch,
                  onValueChange = onCalibration,
                ),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Column {
              Text(
                stringResource(R.string.calibration_title),
                style = MaterialTheme.typography.subtitle2,
              )
              Text(
                stringResource(
                  if (state.calibrated) R.string.calibrated else R.string.uncalibrated
                ),
                style = MaterialTheme.typography.caption,
              )
            }
            Switch(checked = state.calibrated, onCheckedChange = null, enabled = editable)
          }
        }
        item {
          Button(
            onClick = onRun,
            enabled = editable && state.ready,
            modifier = Modifier.fillMaxWidth(),
          ) {
            Text(stringResource(R.string.run))
          }
        }
        state.runTotalMs?.let { totalMs ->
          item {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
              Text(
                stringResource(
                  R.string.run_summary,
                  totalMs,
                  state.runTokenCount,
                  state.answers.size,
                ),
                style = MaterialTheme.typography.subtitle1,
              )
              Text(
                stringResource(R.string.token_count_hint),
                style = MaterialTheme.typography.caption,
              )
            }
          }
        }
        if (state.answers.isNotEmpty()) {
          item {
            Text(stringResource(R.string.results_title), style = MaterialTheme.typography.h6)
          }
        }
        items(state.answers, key = { it.questionId }) { result -> AnswerCard(result) }
      }
    }
  }
}

@Composable
private fun StatusHeader(
  state: UiState,
  onAccelerator: (LayaEngine.Backend) -> Unit,
  onDownload: () -> Unit,
) {
  Column(
    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Text(stringResource(state.statusMessage), style = MaterialTheme.typography.subtitle1)
    state.launchToReadyMs?.let {
      Text(stringResource(R.string.launch_to_ready, it), style = MaterialTheme.typography.caption)
    }
    if (state.busy) {
      LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    }
    state.errorMessage?.let { message ->
      Text(stringResource(R.string.error_message, message), color = MaterialTheme.colors.error)
      state.fallback?.let { fallback ->
        OutlinedButton(onClick = { onAccelerator(fallback) }, enabled = !state.busy) {
          Text(
            stringResource(
              if (fallback == LayaEngine.Backend.GPU) R.string.use_gpu else R.string.use_cpu
            )
          )
        }
      }
    }
    if (state.downloadNeeded) {
      DownloadControls(state, onDownload)
    }
  }
  Divider()
}

/** Download button, progress and failure states shown until the model files are on the device. */
@Composable
private fun DownloadControls(state: UiState, onDownload: () -> Unit) {
  val context = LocalContext.current
  val total = Formatter.formatFileSize(context, state.downloadTotalBytes)
  val done = Formatter.formatFileSize(context, state.downloadedBytes)
  val needed = state.spaceNeededBytes
  val available = state.spaceAvailableBytes
  when {
    state.downloading -> {
      LinearProgressIndicator(
        progress =
          if (state.downloadTotalBytes > 0) {
            (state.downloadedBytes.toDouble() / state.downloadTotalBytes).toFloat()
          } else {
            0f
          },
        modifier = Modifier.fillMaxWidth(),
      )
      Text(
        stringResource(R.string.download_progress, done, total),
        style = MaterialTheme.typography.caption,
      )
    }
    needed != null && available != null -> {
      Text(
        stringResource(
          R.string.download_no_space,
          Formatter.formatFileSize(context, needed),
          Formatter.formatFileSize(context, available),
        ),
        color = MaterialTheme.colors.error,
      )
      Button(
        onClick = onDownload,
        enabled = !state.busy && !state.downloading,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(stringResource(R.string.download_retry))
      }
    }
    else ->
      Button(
        onClick = onDownload,
        enabled = !state.busy && !state.downloading,
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(
          when {
            state.downloadFailed -> stringResource(R.string.download_retry)
            state.downloadedBytes > 0 -> stringResource(R.string.download_resume, done, total)
            else -> stringResource(R.string.download_start, total)
          }
        )
      }
  }
  if (state.meteredNetwork && !state.downloading) {
    Text(
      stringResource(R.string.download_wifi_recommended),
      style = MaterialTheme.typography.caption,
    )
  }
}

@Composable
private fun PresetPicker(selected: Preset, enabled: Boolean, onPreset: (Preset) -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  Box {
    OutlinedButton(
      onClick = { expanded = true },
      enabled = enabled,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Text(stringResource(R.string.preset_selected, stringResource(selected.title)))
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      Preset.entries.forEach { preset ->
        DropdownMenuItem(
          onClick = {
            expanded = false
            onPreset(preset)
          }
        ) {
          Text(stringResource(preset.title))
        }
      }
    }
  }
}

@Composable
private fun AcceleratorPicker(
  selected: LayaEngine.Backend,
  npuAvailable: Boolean,
  enabled: Boolean,
  onAccelerator: (LayaEngine.Backend) -> Unit,
) {
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Text(stringResource(R.string.accelerator_title), style = MaterialTheme.typography.subtitle2)
    Row(modifier = Modifier.fillMaxWidth().selectableGroup()) {
      LayaEngine.Backend.entries.forEach { backend ->
        val choosable = enabled && (backend != LayaEngine.Backend.NPU || npuAvailable)
        Row(
          modifier =
            Modifier.weight(1f)
              .selectable(
                selected = selected == backend,
                enabled = choosable,
                role = Role.RadioButton,
                onClick = { onAccelerator(backend) },
              )
              .padding(vertical = 4.dp),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          RadioButton(selected = selected == backend, onClick = null, enabled = choosable)
          Text(
            stringResource(
              when (backend) {
                LayaEngine.Backend.NPU -> R.string.accelerator_npu
                LayaEngine.Backend.GPU -> R.string.accelerator_gpu
                LayaEngine.Backend.CPU -> R.string.accelerator_cpu
              }
            )
          )
        }
      }
    }
    if (!npuAvailable) {
      Text(stringResource(R.string.npu_unavailable), style = MaterialTheme.typography.caption)
    }
  }
}

@Composable
private fun AnswerCard(result: AnswerUiRow) {
  Card(modifier = Modifier.fillMaxWidth(), elevation = 2.dp) {
    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
      Text(result.instructions, style = MaterialTheme.typography.subtitle1)
      Text(
        when (result.type) {
          "choice" -> stringResource(R.string.result_choice, checkNotNull(result.choice))
          "score" -> stringResource(R.string.result_score, checkNotNull(result.score))
          else ->
            stringResource(R.string.result_true_probability, checkNotNull(result.trueProbability))
        },
        style = MaterialTheme.typography.h6,
        color = MaterialTheme.colors.primary,
      )
      if (result.type == "score") {
        Text(stringResource(R.string.score_legend), style = MaterialTheme.typography.caption)
      }
      result.probabilities.forEach { ProbabilityBar(it) }
      Divider()
      Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
          stringResource(R.string.result_confidence, result.confidence),
          style = MaterialTheme.typography.caption,
        )
        Text(
          stringResource(R.string.result_timing, result.totalMs),
          style = MaterialTheme.typography.caption,
        )
      }
    }
  }
}

@Composable
private fun ProbabilityBar(option: ProbabilityUiRow) {
  val label =
    when {
      option.labelResource != null -> stringResource(option.labelResource)
      option.scoreLevel != null -> stringResource(R.string.score_level, option.scoreLevel)
      else -> option.label
    }
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
      Text(label, style = MaterialTheme.typography.body2, modifier = Modifier.weight(1f))
      Text(
        stringResource(R.string.option_probability, option.probability * 100.0),
        style = MaterialTheme.typography.body2,
      )
    }
    LinearProgressIndicator(
      progress = option.probability.toFloat().coerceIn(0f, 1f),
      modifier = Modifier.fillMaxWidth(),
    )
    option.description
      ?.takeIf { it.isNotEmpty() }
      ?.let { Text(it, style = MaterialTheme.typography.caption) }
  }
}
