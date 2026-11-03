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

package com.google.ai.edge.examples.receipt_to_json

import android.os.Build
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val Ink = Color(0xFF1B1B1F)
private val Accent = Color(0xFF1A73E8)
private val Muted = Color(0xFF5F6368)
private val Paper = Color(0xFFF6F7F9)
private val Good = Color(0xFF188038)
private val Bad = Color(0xFFB3261E)

/** The photo and the answer sit side by side in a panel of this height. */
private val PANEL_HEIGHT = 400.dp

/** The screen: the photo beside its JSON, the keys to extract, the photo sources and Extract. */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun ReceiptScreen(
  ui: UiState,
  onPickPhoto: () -> Unit,
  onCamera: () -> Unit,
  onExample: () -> Unit,
  onKeys: (String) -> Unit,
  onExtract: () -> Unit,
  onStop: () -> Unit,
  onDownload: () -> Unit,
  onCopy: (String) -> Unit,
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
        verticalArrangement = Arrangement.spacedBy(10.dp),
      ) {
        Header(ui)
        ModelPanel(ui, onDownload)
        Row(
          Modifier.fillMaxWidth().height(PANEL_HEIGHT),
          horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
          PhotoPanel(ui, Modifier.weight(0.42f).fillMaxHeight())
          AnswerPanel(ui, onCopy, Modifier.weight(0.58f).fillMaxHeight())
        }
        OutlinedTextField(
          value = ui.keysText,
          onValueChange = onKeys,
          enabled = !ui.reading,
          label = { Text("Keys to extract (comma-separated; a list: items[name, price])") },
          textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
          maxLines = 3,
          modifier = Modifier.fillMaxWidth().testTag("keys"),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          OutlinedButton(onClick = onPickPhoto, enabled = !ui.reading) { Text("Photo") }
          OutlinedButton(onClick = onCamera, enabled = !ui.reading) { Text("Camera") }
          OutlinedButton(onClick = onExample, enabled = !ui.reading) { Text("Example") }
        }
        ExtractButton(ui, onExtract, onStop)
        ui.message?.let { msg ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(msg, color = Bad, modifier = Modifier.weight(1f), fontSize = 14.sp)
            TextButton(onClick = onDismissMessage) { Text("OK") }
          }
        }
        Text(
          "${Build.MODEL} · Xiaomi-OCR-0 on the CPU · LiteRT-LM ${BuildConfig.LITERTLM_VERSION}",
          fontSize = 12.sp,
          color = Muted,
          maxLines = 1,
          modifier = Modifier.testTag("device"),
        )
        Spacer(Modifier.height(16.dp))
      }
    }
  }
}

@Composable
private fun Header(ui: UiState) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    Text(
      "Receipt to JSON",
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
        ui.reading -> Accent to "Reading"
        else ->
          when (ui.reader) {
            is ReaderState.Ready -> Good to "Ready"
            ReaderState.Preparing -> Color(0xFFF9AB00) to "Preparing"
            is ReaderState.Failed -> Bad to "Not ready"
            else -> Muted to "Idle"
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
}

@Composable
private fun ModelPanel(ui: UiState, onDownload: () -> Unit) {
  when (val m = ui.model) {
    is ModelState.Missing ->
      Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
          "The app reads receipts on the phone, offline, with Xiaomi-OCR-0. Its model file " +
            "(%.1f GB) downloads once from Hugging Face.".format(m.bytes / 1e9),
          fontSize = 14.sp,
          color = Ink,
        )
        Button(onClick = onDownload, modifier = Modifier.testTag("download")) {
          Text("Download (%.1f GB) and load".format(m.bytes / 1e9))
        }
      }
    is ModelState.Downloading -> {
      Text("Downloading %.2f / %.2f GB".format(m.received / 1e9, m.total / 1e9), fontSize = 14.sp)
      LinearProgressIndicator(
        progress = { if (m.total > 0) m.received.toFloat() / m.total else 0f },
        modifier = Modifier.fillMaxWidth(),
      )
    }
    ModelState.Verifying -> Text("Checking the download (SHA-256)...", fontSize = 14.sp)
    is ModelState.Error ->
      Column {
        Text(m.message, color = Bad, fontSize = 14.sp)
        Button(onClick = onDownload) { Text("Try again") }
      }
    ModelState.Checking -> Text("Looking for the model file...", fontSize = 14.sp)
    ModelState.Ready ->
      when (val r = ui.reader) {
        // Only before the first read: later Engines are made while an answer shows.
        ReaderState.Preparing ->
          if (!ui.reading && ui.answer == null)
            Text("Loading the model...", fontSize = 13.sp, color = Muted)
        is ReaderState.Failed -> Text(r.message, color = Bad, fontSize = 14.sp)
        else -> {}
      }
  }
}

/** The whole photo, fitted into its half of the panel. */
@Composable
private fun PhotoPanel(ui: UiState, modifier: Modifier) {
  val photo = ui.photo
  Box(
    modifier
      .clip(RoundedCornerShape(8.dp))
      .background(Paper)
      .border(1.dp, Color(0xFFDADCE0), RoundedCornerShape(8.dp))
      .testTag("photo"),
    contentAlignment = Alignment.Center,
  ) {
    if (photo == null) {
      Text(
        "Take or choose a photo of a receipt.",
        color = Muted,
        fontSize = 14.sp,
        modifier = Modifier.padding(12.dp),
      )
    } else {
      val bitmap = remember(photo) { photo.asImageBitmap() }
      Image(
        bitmap,
        contentDescription = ui.photoName,
        contentScale = ContentScale.Fit,
        modifier = Modifier.fillMaxSize(),
      )
    }
  }
}

/** The answer as the model writes it (JSON), with the time it took. */
@Composable
private fun AnswerPanel(ui: UiState, onCopy: (String) -> Unit, modifier: Modifier) {
  val a = ui.answer
  Column(
    modifier
      .clip(RoundedCornerShape(8.dp))
      .background(Paper)
      .padding(10.dp)
      .testTag("answer_panel"),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Text("JSON", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Muted)
    // While the answer streams, its last lines stay in view.
    val scroll = rememberScrollState()
    LaunchedEffect(a?.text?.length) { if (a != null && !a.done) scroll.scrollTo(scroll.maxValue) }
    Box(Modifier.weight(1f).fillMaxWidth().verticalScroll(scroll)) {
      if (a == null) {
        Text(
          "Tap Extract: the keys below come back here as JSON.",
          color = Muted,
          fontSize = 13.sp,
        )
      } else {
        SelectionContainer {
          Text(
            a.text,
            fontFamily = FontFamily.Monospace,
            fontSize = 12.sp,
            lineHeight = 16.sp,
            color = Ink,
            modifier = Modifier.testTag("answer"),
          )
        }
      }
    }
    val line =
      when {
        a == null -> null
        a.error != null -> "Stopped: ${a.error}"
        !a.done -> "Reading… %.1f s".format(ui.elapsedMs / 1000.0)
        else -> {
          val keys =
            a.keysFound?.let { "$it of ${a.keys.size} keys" } ?: "no JSON object in the answer"
          val rows = a.rows.joinToString("") { (k, n) -> " · $n $k" }
          "%.1f s · %s%s".format((a.totalMs ?: 0) / 1000.0, keys, rows)
        }
      }
    if (line != null) {
      Text(
        line,
        fontSize = 13.sp,
        color = if (a?.error != null || (a?.done == true && a.keysFound == null)) Bad else Ink,
        fontWeight = FontWeight.Medium,
        modifier = Modifier.testTag("time"),
      )
    }
    if (a != null && a.done && a.text.isNotEmpty()) {
      TextButton(onClick = { onCopy(Kie.firstJsonObject(a.text) ?: a.text) }) { Text("Copy JSON") }
    }
  }
}

@Composable
private fun ExtractButton(ui: UiState, onExtract: () -> Unit, onStop: () -> Unit) {
  if (ui.reading) {
    OutlinedButton(
      onClick = onStop,
      enabled = !ui.stopping,
      modifier = Modifier.fillMaxWidth().testTag("stop"),
    ) {
      Text(if (ui.stopping) "Stopping…" else "Stop")
    }
  } else {
    Button(
      onClick = onExtract,
      enabled = ui.model is ModelState.Ready && ui.photo != null,
      modifier = Modifier.fillMaxWidth().height(52.dp).testTag("extract"),
    ) {
      Text("Extract", fontSize = 17.sp)
    }
  }
}
