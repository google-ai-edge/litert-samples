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

package com.google.ai.edge.examples.click_or_code

import android.os.Build
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
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
private val Mark = Color(0xFFE5004F)
private val CodeBack = Color(0xFF14171F)
private val CodeInk = Color(0xFFE6EDF3)

/** The screen: the screenshot, the request, Decide, and the next step: a point or a command. */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun StepScreen(
  ui: UiState,
  onPick: () -> Unit,
  onExample: (Example) -> Unit,
  onRequest: (String) -> Unit,
  onDecide: () -> Unit,
  onStop: () -> Unit,
  onDownload: () -> Unit,
  onCopy: (String) -> Unit,
  onDismissMessage: () -> Unit,
) {
  MaterialTheme(colorScheme = lightColorScheme(primary = Accent, onSurface = Ink)) {
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
        ScreenshotPanel(ui)
        SourceRow(ui, onPick, onExample)
        OutlinedTextField(
          value = ui.request,
          onValueChange = onRequest,
          enabled = !ui.deciding,
          label = { Text("What should happen on this screen?") },
          textStyle = MaterialTheme.typography.bodyLarge,
          maxLines = 3,
          modifier = Modifier.fillMaxWidth().testTag("request"),
        )
        if (ui.requestChips.isNotEmpty()) RequestChips(ui, onRequest)
        DecideButton(ui, onDecide, onStop)
        ui.message?.let { msg ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
              msg,
              color = Bad,
              modifier = Modifier.weight(1f).testTag("message"),
              fontSize = 14.sp,
            )
            TextButton(onClick = onDismissMessage) { Text("OK") }
          }
        }
        ui.decision?.let { DecisionPanel(it, ui, onCopy) }
        Text(
          "${Build.MODEL} · Mage-VL-Cowork · text GPU, vision CPU · LiteRT-LM " +
            BuildConfig.LITERTLM_VERSION,
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
      "Click or code",
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
        ui.deciding -> Accent to "Deciding"
        else ->
          when (ui.engine) {
            is EngineState.Ready -> Good to "Ready"
            EngineState.Preparing -> Color(0xFFF9AB00) to "Preparing"
            is EngineState.Failed -> Bad to "Not ready"
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
          "The app decides the next step on the phone, offline, with Mage-VL-Cowork. Its model " +
            "file (%.1f GB) downloads once from Hugging Face.".format(m.bytes / 1e9),
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
      when (val r = ui.engine) {
        // Only before the first step: later Conversations are made while a decision shows.
        EngineState.Preparing ->
          if (!ui.deciding && ui.decision == null) {
            Text("Loading the model...", fontSize = 13.sp, color = Muted)
          }
        is EngineState.Failed ->
          Text(r.message, color = Bad, fontSize = 14.sp, modifier = Modifier.testTag("engine_error"))
        else -> {}
      }
  }
}

/** The screenshot, at its own shape, with the click's mark drawn on it. */
@Composable
private fun ScreenshotPanel(ui: UiState) {
  val photo = ui.photo
  if (photo == null) {
    Box(
      Modifier.fillMaxWidth()
        .height(150.dp)
        .clip(RoundedCornerShape(8.dp))
        .background(Paper)
        .border(1.dp, Color(0xFFDADCE0), RoundedCornerShape(8.dp))
        .testTag("screenshot"),
      contentAlignment = Alignment.Center,
    ) {
      Text("Choose a screenshot or an example.", color = Muted, fontSize = 14.sp)
    }
    return
  }
  val bitmap = remember(photo) { photo.asImageBitmap() }
  val ratio = ui.photoWidth.toFloat() / ui.photoHeight
  val portrait = ratio < 1f
  Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
    Box(
      (if (portrait) Modifier.height(300.dp).aspectRatio(ratio)
        else Modifier.fillMaxWidth().aspectRatio(ratio))
        .clip(RoundedCornerShape(6.dp))
        .border(1.dp, Color(0xFFDADCE0), RoundedCornerShape(6.dp))
        .testTag("screenshot")
    ) {
      Image(
        bitmap,
        contentDescription = ui.photoName,
        contentScale = ContentScale.FillBounds,
        modifier = Modifier.fillMaxSize(),
      )
      val call = ui.decision?.takeIf { it.done && !it.stopped }?.reply?.first
      val points = listOfNotNull(call?.coordinate, call?.start, call?.end)
      if (points.isNotEmpty() && call?.isCode != true) MarkOverlay(points, ui.decision?.text ?: "")
    }
  }
}

/** A ring that closes on each point (fractions of the whole screenshot, 0-999 from the model). */
@Composable
private fun MarkOverlay(points: List<Pair<Double, Double>>, key: String) {
  val grow = remember(key) { Animatable(0f) }
  LaunchedEffect(key) { grow.animateTo(1f, tween(450)) }
  Canvas(Modifier.fillMaxSize().testTag("mark")) {
    val r = 11.dp.toPx()
    for (p in points) {
      val c =
        Offset(
          (p.first.coerceIn(0.0, 999.0) / 999.0 * size.width).toFloat(),
          (p.second.coerceIn(0.0, 999.0) / 999.0 * size.height).toFloat(),
        )
      val outer = r * (3f - 2f * grow.value)
      drawCircle(Color.White, radius = outer + 1.5.dp.toPx(), center = c, style = Stroke(5.dp.toPx()))
      drawCircle(Mark, radius = outer, center = c, style = Stroke(3.dp.toPx()))
      drawCircle(Color.White, radius = 4.5.dp.toPx(), center = c)
      drawCircle(Mark, radius = 3.dp.toPx(), center = c)
    }
  }
}

@Composable
private fun SourceRow(ui: UiState, onPick: () -> Unit, onExample: (Example) -> Unit) {
  Row(
    Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
    horizontalArrangement = Arrangement.spacedBy(8.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    OutlinedButton(
      onClick = onPick,
      enabled = !ui.deciding,
      modifier = Modifier.testTag("pick"),
    ) {
      Text("Screenshot…")
    }
    for (e in ui.examples) {
      Chip(e.name, enabled = !ui.deciding, tag = "example_${e.id}") { onExample(e) }
    }
  }
}

@Composable
private fun RequestChips(ui: UiState, onRequest: (String) -> Unit) {
  Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
    ui.requestChips.forEachIndexed { i, r ->
      Chip(r, enabled = !ui.deciding, tag = "chip_$i") { onRequest(r) }
    }
  }
}

@Composable
private fun Chip(text: String, enabled: Boolean, tag: String, onClick: () -> Unit) {
  Box(
    Modifier.clip(RoundedCornerShape(16.dp))
      .background(Paper)
      .border(1.dp, Color(0xFFDADCE0), RoundedCornerShape(16.dp))
      .clickable(enabled = enabled, onClick = onClick)
      .padding(horizontal = 12.dp, vertical = 7.dp)
      .testTag(tag)
  ) {
    Text(text, fontSize = 13.sp, color = Ink)
  }
}

@Composable
private fun DecideButton(ui: UiState, onDecide: () -> Unit, onStop: () -> Unit) {
  if (ui.deciding) {
    OutlinedButton(
      onClick = onStop,
      enabled = !ui.stopping,
      modifier = Modifier.fillMaxWidth().height(52.dp).testTag("stop"),
    ) {
      Text(
        if (ui.stopping) "Stopping…" else "Stop · %.0f s".format(ui.elapsedMs / 1000.0),
        fontSize = 16.sp,
      )
    }
  } else {
    Button(
      onClick = onDecide,
      enabled = ui.model is ModelState.Ready && ui.photo != null && ui.request.isNotBlank(),
      modifier = Modifier.fillMaxWidth().height(52.dp).testTag("decide"),
    ) {
      Text("Decide", fontSize = 17.sp)
    }
  }
}

/** The next step: the word CLICK or CODE, the model's Action sentence, the point or the command. */
@Composable
private fun DecisionPanel(d: Decision, ui: UiState, onCopy: (String) -> Unit) {
  Column(
    Modifier.fillMaxWidth()
      .clip(RoundedCornerShape(8.dp))
      .background(Paper)
      .padding(12.dp)
      .testTag("decision"),
    verticalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    val call = d.reply.first
    if (!d.done) {
      val phase =
        when {
          !d.hasText && d.inference == 2 -> "The model asked for a screenshot. Inference 2: reading the screen…"
          !d.hasText -> "Inference 1: reading the screen…"
          call == null -> "Inference ${d.inference}: thinking…"
          else -> "Inference ${d.inference}: writing the step…"
        }
      Text(phase, fontSize = 15.sp, fontWeight = FontWeight.Medium, color = Ink, modifier = Modifier.testTag("phase"))
      if (d.text.isNotEmpty()) {
        Text(
          d.reply.thinking.takeLast(170),
          fontSize = 12.sp,
          lineHeight = 16.sp,
          color = Muted,
          maxLines = 3,
          modifier = Modifier.testTag("thinking_tail"),
        )
      }
    } else {
      when {
        d.error != null ->
          Text("Stopped: ${d.error}", color = Bad, fontSize = 14.sp, modifier = Modifier.testTag("error"))
        d.stopped -> Text("Stopped.", color = Muted, fontSize = 14.sp, modifier = Modifier.testTag("stopped"))
        call == null ->
          Text(
            if (d.cut) "The reply reached the file's ${Step.FILE_TOKENS} tokens before it gave a step."
            else "The reply has no step in it.",
            color = Bad,
            fontSize = 14.sp,
            modifier = Modifier.testTag("no_step"),
          )
        else -> Answer(call, d, onCopy)
      }
      StagesLine(d)
      Thinking(d)
    }
  }
}

@Composable
private fun Answer(call: ToolCall, d: Decision, onCopy: (String) -> Unit) {
  Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
    val word = if (call.isCode) "CODE" else (call.action ?: call.name).replace('_', ' ').uppercase()
    Box(
      Modifier.clip(RoundedCornerShape(6.dp))
        .background(if (call.isCode) CodeBack else Mark)
        .padding(horizontal = 10.dp, vertical = 4.dp)
    ) {
      Text(
        word,
        color = Color.White,
        fontSize = 17.sp,
        fontWeight = FontWeight.Bold,
        modifier = Modifier.testTag("route"),
      )
    }
    if (!call.isCode) {
      call.coordinate?.let { c ->
        val (x, y) = Step.toPixels(c, d.width, d.height)
        Text("x $x, y $y of ${d.width} × ${d.height}", fontSize = 13.sp, color = Muted, modifier = Modifier.testTag("point"))
      }
    }
  }
  d.reply.actionLine?.let {
    Text(it, fontSize = 17.sp, lineHeight = 22.sp, color = Ink, fontWeight = FontWeight.Medium, modifier = Modifier.testTag("action"))
  }
  if (call.isCode && call.command != null) {
    SelectionContainer {
      Text(
        call.command,
        fontFamily = FontFamily.Monospace,
        fontSize = 14.sp,
        lineHeight = 19.sp,
        color = CodeInk,
        modifier =
          Modifier.fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(CodeBack)
            .padding(10.dp)
            .testTag("command"),
      )
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
      Text("Shown, not run.", fontSize = 12.sp, color = Muted, modifier = Modifier.weight(1f))
      TextButton(onClick = { onCopy(call.command) }, modifier = Modifier.testTag("copy")) { Text("Copy") }
    }
  } else {
    val extra =
      listOfNotNull(
          call.text?.let { "type: $it" },
          call.keys.takeIf { it.isNotEmpty() }?.let { "keys: ${it.joinToString("+")}" },
        )
        .joinToString("\n")
    if (extra.isNotEmpty()) {
      Text(extra, fontFamily = FontFamily.Monospace, fontSize = 13.sp, color = Ink, modifier = Modifier.testTag("args"))
    }
  }
  if (d.reply.calls.size > 1) {
    Text("+ ${d.reply.calls.size - 1} more in the same step", fontSize = 12.sp, color = Muted)
  }
}

@Composable
private fun StagesLine(d: Decision) {
  val total = d.totalMs?.let { "%.1f s".format(it / 1000.0) } ?: return
  val head = "${d.runs.size} inference${if (d.runs.size == 1) "" else "s"} · $total in all"
  val lines =
    d.runs.mapIndexed { i, run ->
      val parts =
        listOfNotNull(
          "%.1f s".format(run.totalS),
          run.visionS?.let { "vision %.1f s".format(it) },
          run.prefillS?.let { "prefill %.1f s".format(it) },
          run.decodeS?.let { "decode %.1f s (%d tokens)".format(it, run.decodeTokens ?: 0) },
        )
      "${i + 1}: " + parts.joinToString(" · ")
    }
  Text(
    (listOf(head) + lines).joinToString("\n"),
    fontSize = 12.sp,
    lineHeight = 16.sp,
    color = Muted,
    modifier = Modifier.testTag("stages"),
  )
}

/** The model's reasoning, folded. */
@Composable
private fun Thinking(d: Decision) {
  val reasoning = d.runs.map { it.thinking }.ifEmpty { listOf(d.reply.thinking) }.filter { it.isNotEmpty() }.joinToString("\n\n")
  if (reasoning.isEmpty()) return
  var open by remember(d.text) { mutableStateOf(false) }
  TextButton(onClick = { open = !open }, modifier = Modifier.testTag("thinking_toggle")) {
    Text(if (open) "Hide the reasoning" else "Show the reasoning", fontSize = 13.sp)
  }
  if (open) {
    SelectionContainer {
      Text(
        reasoning,
        fontSize = 13.sp,
        lineHeight = 18.sp,
        color = Ink,
        modifier = Modifier.heightIn(max = 400.dp).verticalScroll(rememberScrollState()).testTag("thinking"),
      )
    }
  }
}
