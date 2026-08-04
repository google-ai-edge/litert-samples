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

package com.google.ai.edge.examples.text_to_speech_streaming.view

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.text_to_speech_streaming.MainViewModel
import com.google.ai.edge.examples.text_to_speech_streaming.Metrics
import com.google.ai.edge.examples.text_to_speech_streaming.R
import com.google.ai.edge.examples.text_to_speech_streaming.UiState
import java.util.Locale

/**
 * The demo screen: free text input, voice picker, Speak/Stop, a streaming indicator, and the live
 * metrics card (TTFA / RTF / model size) that is the point of the demo.
 */
@Composable
fun TtsScreen(
  uiState: UiState,
  onSpeak: (String) -> Unit,
  onStop: () -> Unit,
  onSelectVoice: (Int) -> Unit,
  modifier: Modifier = Modifier,
) {
  var text by remember { mutableStateOf(MainViewModel.DEFAULT_TEXT) }
  Scaffold(
    modifier = modifier.statusBarsPadding(),
    topBar = {
      TopAppBar(
        backgroundColor = MaterialTheme.colors.secondary,
        title = { Text(text = stringResource(R.string.app_name), color = Color.White) },
      )
    },
  ) { padding ->
    Column(
      modifier =
        Modifier.fillMaxWidth().padding(padding).padding(16.dp).verticalScroll(rememberScrollState())
    ) {
      OutlinedTextField(
        value = text,
        onValueChange = { text = it },
        label = { Text(text = stringResource(R.string.text_hint)) },
        modifier = Modifier.fillMaxWidth(),
        minLines = 3,
      )
      Spacer(modifier = Modifier.height(8.dp))
      Row(verticalAlignment = Alignment.CenterVertically) {
        Button(onClick = { onSpeak(text) }, enabled = uiState.isModelReady) {
          Text(text = stringResource(R.string.action_speak))
        }
        Spacer(modifier = Modifier.width(8.dp))
        if (uiState.isSpeaking) {
          OutlinedButton(onClick = onStop) { Text(text = stringResource(R.string.action_stop)) }
          Spacer(modifier = Modifier.width(12.dp))
          StreamingIndicator()
        } else {
          VoicePicker(uiState.voices, uiState.selectedVoice, onSelectVoice)
        }
      }
      Spacer(modifier = Modifier.height(12.dp))
      uiState.metrics?.let { metrics ->
        MetricsCard(metrics, uiState.modelMegabytes)
        Spacer(modifier = Modifier.height(8.dp))
      }
      Text(
        text = uiState.errorMessage ?: uiState.statusMessage,
        fontSize = 14.sp,
        color = if (uiState.errorMessage != null) MaterialTheme.colors.error else Color.Gray,
      )
    }
  }
}

/** The "small x fast" readout: TTFA and RTF headline the card, model size proves the "small". */
@Composable
private fun MetricsCard(metrics: Metrics, modelMegabytes: Double) {
  Card(elevation = 2.dp, modifier = Modifier.fillMaxWidth()) {
    Column(modifier = Modifier.padding(12.dp)) {
      Row(horizontalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxWidth()) {
        Metric(
          label = stringResource(R.string.metric_ttfa),
          value = if (metrics.ttfaMs > 0) "${metrics.ttfaMs}" else "–",
          unit = "ms",
        )
        Metric(
          label = stringResource(R.string.metric_rtf),
          value = if (metrics.rtf > 0) String.format(Locale.US, "%.3f", metrics.rtf) else "–",
          unit = "×",
        )
        Metric(
          label = stringResource(R.string.metric_model),
          value = String.format(Locale.US, "%.0f", modelMegabytes),
          unit = "MB",
        )
      }
      Spacer(modifier = Modifier.height(8.dp))
      Text(
        text =
          stringResource(
            R.string.metric_detail,
            metrics.audioSeconds,
            metrics.synthMs / 1000.0,
            metrics.sentencesDone,
            metrics.sentencesTotal,
          ),
        fontSize = 12.sp,
        color = Color.Gray,
      )
    }
  }
}

@Composable
private fun Metric(label: String, value: String, unit: String) {
  Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Row(verticalAlignment = Alignment.Bottom) {
      Text(text = value, fontSize = 28.sp, fontWeight = FontWeight.Bold)
      Spacer(modifier = Modifier.width(2.dp))
      Text(text = unit, fontSize = 14.sp, modifier = Modifier.padding(bottom = 3.dp))
    }
    Text(text = label, fontSize = 12.sp, color = Color.Gray)
  }
}

@Composable
private fun VoicePicker(voices: List<String>, selected: Int, onSelect: (Int) -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  OutlinedButton(onClick = { expanded = true }) {
    Text(text = voices.getOrElse(selected) { "voice" })
  }
  DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
    voices.forEachIndexed { index, voice ->
      DropdownMenuItem(
        onClick = {
          onSelect(index)
          expanded = false
        }
      ) {
        Text(text = voice)
      }
    }
  }
}

/** A pulsing label while PCM is being generated and played. */
@Composable
private fun StreamingIndicator() {
  val transition = rememberInfiniteTransition(label = "streaming")
  val alpha by
    transition.animateFloat(
      initialValue = 0.2f,
      targetValue = 1f,
      animationSpec =
        infiniteRepeatable(tween(500, easing = LinearEasing), repeatMode = RepeatMode.Reverse),
      label = "alpha",
    )
  Text(
    text = stringResource(R.string.streaming),
    color = MaterialTheme.colors.secondary.copy(alpha = alpha),
    fontSize = 14.sp,
    fontWeight = FontWeight.Medium,
  )
}
