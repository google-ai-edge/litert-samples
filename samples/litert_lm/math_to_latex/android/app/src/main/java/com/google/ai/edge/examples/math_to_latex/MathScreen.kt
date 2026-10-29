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

package com.google.ai.edge.examples.math_to_latex

import android.graphics.RectF
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.max
import kotlin.math.min

private val Ink = Color(0xFF1B1B1F)
private val Accent = Color(0xFF1A73E8)
private val Muted = Color(0xFF5F6368)
private val Paper = Color(0xFFF6F7F9)
private val PHOTO_MAX_HEIGHT = 400.dp

/**
 * While a read runs or its answer shows, the photo shrinks so the answer fits on the same screen.
 */
private val PHOTO_SMALL_HEIGHT = 180.dp

/** The screen: the photo with its box, the controls, and the answer. */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun MathScreen(
  ui: UiState,
  onPickPhoto: () -> Unit,
  onCamera: () -> Unit,
  onExample: () -> Unit,
  onBox: (RectF?) -> Unit,
  onMode: (Mode) -> Unit,
  onRead: () -> Unit,
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
        PhotoPanel(ui, onBox)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          OutlinedButton(onClick = onPickPhoto, enabled = !ui.reading) { Text("Photo") }
          OutlinedButton(onClick = onCamera, enabled = !ui.reading) { Text("Camera") }
          OutlinedButton(onClick = onExample, enabled = !ui.reading) { Text("Example") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          Mode.entries.forEach { m ->
            FilterChip(
              selected = ui.mode == m,
              onClick = { onMode(m) },
              enabled = !ui.reading,
              label = { Text(m.label) },
              modifier = Modifier.testTag("mode_${m.name}"),
            )
          }
        }
        ReadButton(ui, onRead, onStop)
        ui.message?.let { msg ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(msg, color = Color(0xFFB3261E), modifier = Modifier.weight(1f), fontSize = 14.sp)
            TextButton(onClick = onDismissMessage) { Text("OK") }
          }
        }
        ui.answer?.let { AnswerPanel(it, ui, onCopy) }
        Spacer(Modifier.height(24.dp))
      }
    }
  }
}

@Composable
private fun Header(ui: UiState) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    // One line whatever the status says: a wrapped title would move everything below it.
    Text(
      "Math to LaTeX",
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
          when (val r = ui.reader) {
            is ReaderState.Ready -> Color(0xFF188038) to "Ready"
            ReaderState.Preparing -> Color(0xFFF9AB00) to "Preparing"
            is ReaderState.Failed -> Color(0xFFB3261E) to "Not ready"
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
          "The app reads on the phone, offline, with MiniCPM-OCR-1.0. Its model file " +
            "(%.1f GB) downloads once from Hugging Face.".format(m.bytes / 1e9),
          fontSize = 14.sp,
          color = Ink,
        )
        Button(onClick = onDownload) { Text("Download (%.1f GB) and load".format(m.bytes / 1e9)) }
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
        Text(m.message, color = Color(0xFFB3261E), fontSize = 14.sp)
        Button(onClick = onDownload) { Text("Try again") }
      }
    ModelState.Checking -> Text("Looking for the model file...", fontSize = 14.sp)
    ModelState.Ready ->
      when (val r = ui.reader) {
        // Only before the first read: later Engines are made while an answer shows (the header says
        // so).
        ReaderState.Preparing ->
          if (!ui.reading && ui.answer == null)
            Text("Loading the model...", fontSize = 13.sp, color = Muted)
        is ReaderState.Failed -> Text(r.message, color = Color(0xFFB3261E), fontSize = 14.sp)
        else -> {}
      }
  }
}

/** The photo, fitted into the panel; drag on it to draw the box to read. */
@Composable
private fun PhotoPanel(ui: UiState, onBox: (RectF?) -> Unit) {
  val photo = ui.photo
  if (photo == null) {
    Box(
      Modifier.fillMaxWidth().height(180.dp).clip(RoundedCornerShape(12.dp)).background(Paper),
      contentAlignment = Alignment.Center,
    ) {
      Text(
        "Take or choose a photo of a page,\nthen drag a box around a formula.",
        color = Muted,
        fontSize = 14.sp,
      )
    }
    return
  }
  val bitmap = remember(photo) { photo.asImageBitmap() }
  var dragStart by remember { mutableStateOf<Offset?>(null) }
  var dragEnd by remember { mutableStateOf<Offset?>(null) }
  BoxWithConstraints(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
    // Fit the photo into the width and the height limit, keeping its aspect ratio.
    val aspect = photo.width.toFloat() / photo.height
    val maxH = if (ui.reading || ui.answer != null) PHOTO_SMALL_HEIGHT else PHOTO_MAX_HEIGHT
    val w = min(maxWidth.value, maxH.value * aspect)
    val h = w / aspect
    Box(
      Modifier.size(w.dp, h.dp)
        .clip(RoundedCornerShape(8.dp))
        .border(1.dp, Color(0xFFDADCE0), RoundedCornerShape(8.dp))
        .testTag("photo")
        .pointerInput(ui.reading) {
          if (ui.reading) return@pointerInput
          // The box runs from where the finger went down to where it lifts.
          awaitEachGesture {
            val down = awaitFirstDown()
            dragStart = down.position
            dragEnd = down.position
            var moved = false
            while (true) {
              val change = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
              if (!change.pressed) break
              dragEnd = change.position
              if ((change.position - down.position).getDistance() > viewConfiguration.touchSlop)
                moved = true
              change.consume()
            }
            val a = dragStart
            val b = dragEnd
            val sw = size.width.toFloat()
            val sh = size.height.toFloat()
            if (moved && a != null && b != null && sw > 0 && sh > 0) {
              val l = (min(a.x, b.x) / sw).coerceIn(0f, 1f)
              val t = (min(a.y, b.y) / sh).coerceIn(0f, 1f)
              val r = (max(a.x, b.x) / sw).coerceIn(0f, 1f)
              val btm = (max(a.y, b.y) / sh).coerceIn(0f, 1f)
              onBox(if (r - l > 0.01f && btm - t > 0.01f) RectF(l, t, r, btm) else null)
            }
            dragStart = null
            dragEnd = null
          }
        }
        .drawWithContent {
          drawContent()
          val live =
            dragStart?.let { a ->
              dragEnd?.let { b ->
                RectF(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
              }
            }
          val box =
            live
              ?: ui.box?.let {
                RectF(
                  it.left * size.width,
                  it.top * size.height,
                  it.right * size.width,
                  it.bottom * size.height,
                )
              }
          if (box != null) {
            // Dim what is outside the box, then outline it.
            val shade = Color(0x66000000)
            drawRect(shade, Offset.Zero, Size(size.width, box.top))
            drawRect(shade, Offset(0f, box.bottom), Size(size.width, size.height - box.bottom))
            drawRect(shade, Offset(0f, box.top), Size(box.left, box.height()))
            drawRect(shade, Offset(box.right, box.top), Size(size.width - box.right, box.height()))
            drawRect(
              Accent,
              Offset(box.left, box.top),
              Size(box.width(), box.height()),
              style = Stroke(width = 3.dp.toPx()),
            )
          }
        }
    ) {
      Image(
        bitmap,
        contentDescription = ui.photoName,
        contentScale = ContentScale.FillBounds,
        modifier = Modifier.fillMaxSize(),
      )
    }
  }
  if (!ui.reading && ui.answer == null) {
    Text(
      if (ui.box == null) "Drag a box around what to read (no box = the whole photo)."
      else "Drag again to change the box.",
      fontSize = 12.sp,
      color = Muted,
    )
  }
}

@Composable
private fun ReadButton(ui: UiState, onRead: () -> Unit, onStop: () -> Unit) {
  if (ui.reading) {
    Row(
      verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      OutlinedButton(
        onClick = onStop,
        enabled = !ui.stopping,
        modifier = Modifier.testTag("stop"),
      ) {
        Text("Stop")
      }
      val waiting = ui.reader !is ReaderState.Reading && ui.answer?.text.isNullOrEmpty()
      val what =
        when {
          // A Stop does not end a read at once (see OcrEngine.cancel).
          ui.stopping -> "Stopping (a whole page can take 20 s or more)... "
          waiting -> "Getting the reader ready... "
          else -> "Reading... "
        }
      Text(
        what + "%.1f s".format(ui.elapsedMs / 1000.0),
        color = Muted,
        fontSize = 14.sp,
        modifier = Modifier.testTag("elapsed"),
      )
    }
  } else {
    Button(
      onClick = onRead,
      // After a failed Engine (its message shows above), a tap tries again.
      enabled = ui.photo != null && ui.model is ModelState.Ready,
      modifier = Modifier.fillMaxWidth().height(52.dp).testTag("read"),
      colors = ButtonDefaults.buttonColors(containerColor = Accent),
    ) {
      Text(if (ui.box == null) "Read the photo" else "Read the box", fontSize = 17.sp)
    }
  }
}

@Composable
private fun AnswerPanel(answer: Answer, ui: UiState, onCopy: (String) -> Unit) {
  Column(
    Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Paper).padding(12.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    // Copy and the timing come first, so they stay on the screen above a long answer; the row is
    // there from the
    // start of the read, so nothing moves when the answer ends.
    Row(verticalAlignment = Alignment.CenterVertically) {
      Button(
        onClick = { onCopy(answer.text) },
        enabled = answer.done && answer.text.isNotEmpty(),
        modifier = Modifier.testTag("copy"),
      ) {
        Text(if (answer.mode == Mode.EQUATION) "Copy LaTeX" else "Copy text")
      }
      Spacer(Modifier.width(12.dp))
      val line =
        if (!answer.done) {
          ""
        } else {
          listOfNotNull(
              answer.totalMs?.let { "%.1f s".format(it / 1000.0) },
              answer.firstTextMs?.let { "first text %.1f s".format(it / 1000.0) },
              if (answer.pictures > 1) "${answer.pictures} pictures" else null,
              "offline",
            )
            .joinToString(" · ")
        }
      Text(line, fontSize = 12.sp, color = Muted, modifier = Modifier.testTag("timing"))
    }
    answer.error?.let { Text(it, color = Color(0xFFB3261E), fontSize = 13.sp) }
    // Made when the read starts (the WebView and MathJax load meanwhile), drawn when the answer
    // ends.
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(Color.White)) {
      MathView(if (answer.done) answer.text else "", Modifier.testTag("typeset"))
    }
    SelectionContainer {
      Text(
        answer.text.ifEmpty { " " },
        fontFamily = FontFamily.Monospace,
        fontSize = 14.sp,
        color = Ink,
        modifier = Modifier.fillMaxWidth().testTag("latex"),
      )
    }
  }
}
