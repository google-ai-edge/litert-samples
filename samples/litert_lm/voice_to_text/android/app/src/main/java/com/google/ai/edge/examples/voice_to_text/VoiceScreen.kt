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

package com.google.ai.edge.examples.voice_to_text

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val Ink = Color(0xFF1B1B1F)
private val Accent = Color(0xFF1A73E8)
private val Muted = Color(0xFF5F6368)
private val Paper = Color(0xFFF6F7F9)
private val RecordRed = Color(0xFFD93025)
private val ErrorRed = Color(0xFFB3261E)

/** The screen: the model, the task, the Record button and the transcripts. */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun VoiceScreen(
  ui: UiState,
  onDownload: () -> Unit,
  onLoad: () -> Unit,
  onTask: (Task) -> Unit,
  onKeywords: (String) -> Unit,
  onRecord: () -> Unit,
  onStop: () -> Unit,
  onCancel: () -> Unit,
  onAgain: () -> Unit,
  onCopy: (String) -> Unit,
  onMessageAction: (MessageAction) -> Unit,
  onDismissMessage: () -> Unit,
) {
  MaterialTheme(colorScheme = lightColorScheme(primary = Accent, onSurface = Ink)) {
    // testTagsAsResourceId: the tags below show as resource ids to UI Automator (the README's
    // scripted runs).
    Surface(
      color = Color.White,
      modifier = Modifier.fillMaxSize().semantics { testTagsAsResourceId = true },
    ) {
      Column(
        Modifier.fillMaxSize()
          .windowInsetsPadding(WindowInsets.safeDrawing)
          .verticalScroll(rememberScrollState())
          .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        Header(ui)
        ModelPanel(ui, onDownload, onLoad)
        TaskPanel(ui, onTask, onKeywords)
        RecordPanel(ui, onRecord, onStop, onCancel, onAgain)
        ui.message?.let { msg ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
              msg,
              color = ErrorRed,
              modifier = Modifier.weight(1f).testTag("message"),
              fontSize = 14.sp,
            )
            if (ui.messageAction == MessageAction.OPEN_SETTINGS) {
              TextButton(
                onClick = { onMessageAction(MessageAction.OPEN_SETTINGS) },
                modifier = Modifier.testTag("open_settings"),
              ) {
                Text("Open settings")
              }
            }
            TextButton(onClick = onDismissMessage) { Text("OK") }
          }
        }
        ui.results.forEachIndexed { index, t -> TranscriptCard(t, latest = index == 0, onCopy) }
        Spacer(Modifier.height(24.dp))
      }
    }
  }
}

@Composable
private fun Header(ui: UiState) {
  Column {
    Row(verticalAlignment = Alignment.CenterVertically) {
      // One line whatever the status says: a wrapped title would move everything below it.
      Text(
        "Voice to Text",
        fontSize = 22.sp,
        fontWeight = FontWeight.SemiBold,
        color = Ink,
        maxLines = 1,
        softWrap = false,
        modifier = Modifier.weight(1f),
      )
      val (dot, label) =
        when {
          ui.model !is ModelState.Ready -> Muted to "No model yet"
          ui.recording -> RecordRed to "Listening"
          ui.transcribing -> Accent to "Writing"
          else ->
            when (ui.engine) {
              is EngineState.Ready -> Color(0xFF188038) to "Ready"
              EngineState.Loading,
              EngineState.WarmingUp -> Color(0xFFF9AB00) to "Loading"
              is EngineState.Failed -> ErrorRed to "Not loaded"
              EngineState.Idle -> Muted to "Idle"
            }
        }
      Box(Modifier.size(9.dp).clip(RoundedCornerShape(50)).background(dot))
      Spacer(Modifier.width(6.dp))
      Text(
        "$label · on-device",
        fontSize = 13.sp,
        color = Muted,
        maxLines = 1,
        modifier = Modifier.testTag("status"),
      )
    }
    Text(
      "granite-speech-5.0-2b on the GPU · LiteRT-LM ${BuildConfig.LITERTLM_VERSION}",
      fontSize = 12.sp,
      color = Muted,
      maxLines = 1,
    )
  }
}

@Composable
private fun ModelPanel(ui: UiState, onDownload: () -> Unit, onLoad: () -> Unit) {
  when (val m = ui.model) {
    ModelState.Checking -> Text("Looking for the model file…", fontSize = 14.sp, color = Muted)
    is ModelState.Missing ->
      Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
          "The app writes down speech in English, French, German, Spanish, Portuguese and " +
            "Japanese on the phone, offline, with granite-speech-5.0-2b. Its model file " +
            "(%.1f GB) downloads once from Hugging Face.".format(m.bytes / 1e9),
          fontSize = 14.sp,
          color = Ink,
        )
        Button(onClick = onDownload, modifier = Modifier.testTag("download")) {
          Text("Download (%.1f GB) and load".format(m.bytes / 1e9))
        }
      }
    is ModelState.Paused ->
      Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
          "Downloaded %.1f of %.1f GB so far.".format(m.received / 1e9, m.total / 1e9),
          fontSize = 14.sp,
          color = Ink,
          modifier = Modifier.testTag("paused"),
        )
        Button(onClick = onDownload, modifier = Modifier.testTag("download")) {
          Text("Resume the download")
        }
      }
    is ModelState.Downloading ->
      Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
          "Downloading %.2f of %.2f GB".format(m.received / 1e9, m.total / 1e9),
          fontSize = 14.sp,
          color = Ink,
          modifier = Modifier.testTag("downloading"),
        )
        LinearProgressIndicator(
          progress = { if (m.total > 0) m.received.toFloat() / m.total else 0f },
          modifier = Modifier.fillMaxWidth(),
        )
      }
    ModelState.Verifying ->
      Text("Checking the file's SHA-256…", fontSize = 14.sp, color = Muted)
    is ModelState.Error ->
      Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(m.message, fontSize = 14.sp, color = ErrorRed, modifier = Modifier.testTag("model_error"))
        Button(onClick = onDownload, modifier = Modifier.testTag("download")) {
          Text("Download again")
        }
      }
    ModelState.Ready ->
      when (val e = ui.engine) {
        EngineState.Loading ->
          Text(
            "Loading the model onto the GPU (up to 20 s the first time)…",
            fontSize = 14.sp,
            color = Muted,
          )
        EngineState.WarmingUp ->
          Text("Loading: a first short run (about 10 s)…", fontSize = 14.sp, color = Muted)
        is EngineState.Failed ->
          Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(e.message, fontSize = 14.sp, color = ErrorRed, modifier = Modifier.testTag("engine_error"))
            OutlinedButton(onClick = onLoad, modifier = Modifier.testTag("load")) { Text("Load") }
          }
        else -> {}
      }
  }
}

@Composable
private fun TaskPanel(ui: UiState, onTask: (Task) -> Unit, onKeywords: (String) -> Unit) {
  val idle = !ui.recording && !ui.transcribing
  Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Task.entries.forEach { t ->
        FilterChip(
          selected = ui.task == t,
          onClick = { onTask(t) },
          enabled = idle,
          label = { Text(t.label) },
          modifier = Modifier.testTag("task_${t.name}"),
        )
      }
    }
    OutlinedTextField(
      value = ui.keywords,
      onValueChange = onKeywords,
      enabled = idle,
      singleLine = true,
      label = { Text("Keywords (optional): names or terms, comma-separated") },
      modifier = Modifier.fillMaxWidth().testTag("keywords"),
    )
  }
}

@Composable
private fun RecordPanel(
  ui: UiState,
  onRecord: () -> Unit,
  onStop: () -> Unit,
  onCancel: () -> Unit,
  onAgain: () -> Unit,
) {
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    when {
      ui.recording -> {
        Button(
          onClick = onStop,
          colors = ButtonDefaults.buttonColors(containerColor = RecordRed),
          modifier = Modifier.fillMaxWidth().height(64.dp).testTag("stop"),
        ) {
          Text("■  Stop   ${clock(ui.recordedMs)} / ${clock(ui.maxSeconds * 1000L)}", fontSize = 20.sp)
        }
        LinearProgressIndicator(
          progress = { ui.level.coerceIn(0f, 1f) },
          color = RecordRed,
          modifier = Modifier.fillMaxWidth().testTag("level"),
        )
      }
      ui.transcribing ->
        OutlinedButton(
          onClick = onCancel,
          enabled = !ui.stopping,
          modifier = Modifier.fillMaxWidth().height(64.dp).testTag("cancel"),
        ) {
          Text(if (ui.stopping) "Stopping…" else "Writing…  tap to stop", fontSize = 18.sp)
        }
      else ->
        Button(
          onClick = onRecord,
          colors = ButtonDefaults.buttonColors(containerColor = RecordRed),
          modifier = Modifier.fillMaxWidth().height(64.dp).testTag("record"),
        ) {
          Text("●  Record", fontSize = 20.sp)
        }
    }
    if (ui.hasLastRecording && !ui.recording) {
      TextButton(
        onClick = onAgain,
        enabled = !ui.transcribing,
        modifier = Modifier.testTag("again"),
      ) {
        Text("Run the last recording again as: ${ui.task.label}")
      }
    }
  }
}

@Composable
private fun TranscriptCard(t: Transcript, latest: Boolean, onCopy: (String) -> Unit) {
  Column(
    Modifier.fillMaxWidth()
      .clip(RoundedCornerShape(12.dp))
      .background(if (latest) Paper else Color.White)
      .padding(12.dp)
      .testTag(if (latest) "result_latest" else "result_${t.id}"),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Text(
      buildString {
        append(t.task.label)
        if (t.keywords.isNotEmpty()) append(" · keywords: ${t.keywords.joinToString(", ")}")
        append(" · %.1f s of speech".format(t.seconds))
      },
      fontSize = 13.sp,
      color = Muted,
    )
    SelectionContainer {
      Text(
        when {
          t.text.isNotEmpty() -> t.text
          t.done -> ""
          else -> "…"
        },
        fontSize = if (latest) 19.sp else 16.sp,
        color = Ink,
        lineHeight = if (latest) 26.sp else 22.sp,
        modifier = Modifier.testTag(if (latest) "text_latest" else "text_${t.id}"),
      )
    }
    t.error?.let { Text(it, fontSize = 13.sp, color = ErrorRed) }
    Row(verticalAlignment = Alignment.CenterVertically) {
      Text(
        timing(t),
        fontSize = 12.sp,
        color = Muted,
        modifier = Modifier.weight(1f).testTag(if (latest) "timing_latest" else "timing_${t.id}"),
      )
      if (t.done && t.text.isNotEmpty()) {
        TextButton(onClick = { onCopy(t.text) }) { Text("Copy") }
      }
    }
  }
}

private fun timing(t: Transcript): String =
  buildString {
    if (t.firstTextMs != null) append("first words %.1f s after Stop".format(t.firstTextMs / 1000.0))
    if (t.done && t.totalMs != null) {
      if (isNotEmpty()) append(" · ")
      append(
        when {
          t.cancelled -> "stopped after %.1f s".format(t.totalMs / 1000.0)
          t.error != null -> "failed after %.1f s".format(t.totalMs / 1000.0)
          else -> "done in %.1f s".format(t.totalMs / 1000.0)
        }
      )
    }
    if (isEmpty()) append("writing…")
  }

private fun clock(ms: Long): String {
  val s = ms / 1000
  return "%d:%02d".format(s / 60, s % 60)
}
