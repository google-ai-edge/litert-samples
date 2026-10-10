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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// samples/voice/src/main/kotlin/io/github/johnrocky/hfmodels/samples/voice/VoiceScreen.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.examples.voice_assistant.data.DownloadStatus

private val BG = Color(0xFF0B0F14)
private val CARD = Color(0xFF141B24)
private val FG = Color(0xFFF2F4F7)
private val DIM = Color(0xFF8A97A6)
private val ACCENT = Color(0xFF22D3EE)
private val GREEN = Color(0xFF3FB950)
private val RED = Color(0xFFF85149)
private val LIVE = Color(0xFFE5484D)

/**
 * The one screen, top to bottom: the network state, the status, the three models with their
 * download state, what was heard, the tool calls with the phone's answers, what was said, the
 * time from the end of speech to the first sound, the Load / microphone button, and Android's next
 * alarm. On a metered network one dialog asks before the download starts.
 */
@Composable
fun VoiceScreen(
  ui: VoiceUi,
  onMic: () -> Unit,
  onLoad: () -> Unit,
  onConfirmDownload: () -> Unit,
  onCancelDownload: () -> Unit,
) {
  MaterialTheme(colorScheme = darkColorScheme(background = BG, surface = CARD, primary = ACCENT)) {
    ui.downloadConfirmation?.let { bytes ->
      AlertDialog(
        onDismissRequest = onCancelDownload,
        title = { Text("Download on a metered network?") },
        text = {
          Text(
            "The models still to download are ${size(bytes)}. This network is metered, so the " +
              "download may cost money."
          )
        },
        confirmButton = { Button(onClick = onConfirmDownload) { Text("Download") } },
        dismissButton = { TextButton(onClick = onCancelDownload) { Text("Cancel") } },
      )
    }
    Column(
      Modifier.fillMaxSize()
        .background(BG)
        .safeDrawingPadding()
        .padding(horizontal = 22.dp, vertical = 16.dp)
    ) {
      val network = ui.network.ifEmpty { "?" }.uppercase()
      val airplane = if (ui.airplaneMode) " · AIRPLANE MODE" else ""
      Text(
        "ON-DEVICE · LiteRT · NETWORK: $network$airplane",
        color = DIM,
        fontSize = 11.sp,
        letterSpacing = 1.6.sp,
      )
      Text(ui.status, color = FG, fontSize = 15.sp, modifier = Modifier.padding(top = 10.dp))
      Column(
        Modifier.weight(1f)
          .fillMaxWidth()
          .padding(top = 14.dp)
          .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        if (!ui.ready) {
          Card {
            for (m in ui.models) {
              Text("${m.role} · ${size(m.bytes)}", color = DIM, fontSize = 12.sp)
              val ready = m.state.status == DownloadStatus.READY
              Text(
                "${m.model}: ${m.stateText}",
                color = if (ready) GREEN else FG,
                fontSize = 14.sp,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(bottom = 4.dp),
              )
            }
          }
        }
        if (ui.heard.isNotEmpty()) {
          Card {
            Text("Heard", color = DIM, fontSize = 12.sp)
            Text(ui.heard, color = FG, fontSize = 22.sp, fontWeight = FontWeight.Medium)
          }
        }
        if (ui.tools.isNotEmpty()) {
          Card {
            for (t in ui.tools) {
              Text(
                "${t.icon} ${t.call}",
                color = DIM,
                fontSize = 13.sp,
                fontFamily = FontFamily.Monospace,
              )
              Text(
                t.result,
                color = if (t.result.startsWith("Error")) RED else GREEN,
                fontSize = 16.sp,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(bottom = 4.dp),
              )
            }
          }
        }
        if (ui.reply.isNotEmpty()) {
          Card {
            Text("Said", color = DIM, fontSize = 12.sp)
            Text(ui.reply, color = FG, fontSize = 20.sp)
            ui.modelReply?.let {
              Text(
                "The model said: $it",
                color = DIM,
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 6.dp),
              )
            }
          }
        }
        ui.error?.let { Text(it, color = RED, fontSize = 13.sp) }
      }
      if (ui.replyIn.isNotEmpty()) {
        Text(
          ui.replyIn,
          color = ACCENT,
          fontSize = 34.sp,
          fontWeight = FontWeight.Bold,
          fontFamily = FontFamily.Monospace,
          modifier = Modifier.padding(top = 10.dp),
        )
        Text(ui.breakdown, color = DIM, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
      }
      Box(
        Modifier.fillMaxWidth().padding(vertical = 18.dp),
        contentAlignment = Alignment.Center,
      ) {
        if (ui.ready || ui.listening) {
          Button(
            onClick = onMic,
            shape = CircleShape,
            modifier = Modifier.size(96.dp),
            colors =
              ButtonDefaults.buttonColors(
                containerColor = if (ui.listening) LIVE else ACCENT,
                contentColor = BG,
              ),
          ) {
            Text(
              if (ui.listening) "Stop" else "🎤",
              fontSize = if (ui.listening) 18.sp else 34.sp,
            )
          }
        } else {
          val missing = ui.missingBytes
          val label =
            if (ui.loading) {
              "Loading…"
            } else if (missing > 0) {
              "Download (${size(missing)}) and load"
            } else {
              "Load the models"
            }
          Button(onClick = onLoad, enabled = !ui.loading, shape = RoundedCornerShape(24.dp)) {
            Text(label)
          }
        }
      }
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
          ui.phoneState.lineSequence().firstOrNull().orEmpty(),
          color = DIM,
          fontSize = 13.sp,
          fontFamily = FontFamily.Monospace,
        )
      }
      Spacer(Modifier.height(4.dp))
    }
  }
}

@Composable
private fun Card(content: @Composable () -> Unit) {
  Column(
    Modifier.fillMaxWidth().background(CARD, RoundedCornerShape(18.dp)).padding(16.dp),
    verticalArrangement = Arrangement.spacedBy(4.dp),
  ) {
    content()
  }
}
