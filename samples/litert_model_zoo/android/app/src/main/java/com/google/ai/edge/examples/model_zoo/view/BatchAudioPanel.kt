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

package com.google.ai.edge.examples.model_zoo.view

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.google.ai.edge.examples.model_zoo.MainViewModel
import com.google.ai.edge.examples.model_zoo.R
import com.google.ai.edge.examples.model_zoo.UiState

@Composable
internal fun BatchAudioPanel(
  state: UiState,
  vm: MainViewModel,
  record: () -> Unit,
  chooseWav: () -> Unit,
) {
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    Text(
      "Record up to 12 seconds or choose a WAV file. The model uses its supported audio window.",
      style = MaterialTheme.typography.bodyMedium,
    )
    AudioInputLabel(state)
    if (state.selectedTaskId == "audio-source-separation") {
      if (state.busy) {
        val progress = state.audioProgress
        Text(
          if (progress == null) stringResource(R.string.tiger_preparing)
          else
            stringResource(
              R.string.tiger_stem_progress,
              progress.stem,
              progress.chunk,
              progress.totalChunks,
            )
        )
        LinearProgressIndicator(Modifier.fillMaxWidth())
      }
    }
    Button(
      onClick = { if (state.recording) vm.stopRecording() else record() },
      enabled = !state.busy,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Text(if (state.recording) "Stop and run" else "Record audio")
    }
    OutlinedButton(
      onClick = chooseWav,
      enabled = !state.busy && !state.recording,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Text("Choose WAV and run")
    }
    if (state.recording) Text("Recording: %.1f s".format(state.recordedSeconds))
  }
}

@Composable
internal fun BatchAudioResultPanel(state: UiState, vm: MainViewModel) {
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    if (state.audioSummary.isNotBlank()) SelectionContainer { Text(state.audioSummary) }
    if (state.pitchHz.isNotEmpty()) {
      val maximum = state.pitchHz.maxOrNull()?.coerceAtLeast(1f) ?: 1f
      Text("Pitch (Hz) · confidence shown by line opacity")
      Canvas(Modifier.fillMaxWidth().height(160.dp)) {
        drawLine(Color.Gray, Offset(0f, size.height), Offset(size.width, size.height))
        for (i in 1 until state.pitchHz.size) {
          val confidence = state.pitchConfidence.getOrElse(i) { 1f }.coerceIn(0.15f, 1f)
          drawLine(
            Color(0xFF006A60).copy(alpha = confidence),
            Offset(
              (i - 1).toFloat() / (state.pitchHz.size - 1) * size.width,
              size.height * (1f - state.pitchHz[i - 1] / maximum),
            ),
            Offset(
              i.toFloat() / (state.pitchHz.size - 1) * size.width,
              size.height * (1f - state.pitchHz[i] / maximum),
            ),
            strokeWidth = 2.dp.toPx(),
          )
        }
      }
      Text(
        "0–%.2f s · 0–%.1f Hz".format((state.pitchHz.size - 1) * state.pitchHopSeconds, maximum),
        style = MaterialTheme.typography.bodySmall,
      )
    }
    if (state.audioOutputs.isNotEmpty()) PlaybackProgress(state)
    if (state.playing) OutlinedButton(onClick = vm::stopPlayback) { Text("Stop playback") }
    state.audioOutputs.forEachIndexed { index, name ->
      OutlinedButton(
        onClick = { vm.playAudioOutput(index) },
        enabled = !state.playing && !state.busy,
      ) {
        Text("Play $name")
      }
    }
  }
}
