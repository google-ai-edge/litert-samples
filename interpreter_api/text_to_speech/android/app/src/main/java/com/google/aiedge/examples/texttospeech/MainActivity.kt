/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.texttospeech

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.aiedge.examples.texttospeech.ui.ApplicationTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ApplicationTheme {
                TtsScreen()
            }
        }
    }
}

@Composable
fun TtsScreen(viewModel: MainViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Spacer(Modifier.height(24.dp))
        Text("Kokoro-82M · LiteRT", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Text(
            "First TTS on LiteRT — StyleTTS2 + ISTFTNet on the CPU runtime (XNNPACK). " +
                "Fixed-length preview: reproduces one baked utterance.",
            fontSize = 14.sp,
        )

        ElevatedCard {
            Column(
                Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text("Fixed demo utterance", fontWeight = FontWeight.SemiBold)
                Text("57 phoneme tokens → ~3.7 s @ 24 kHz, baked at export.", fontSize = 12.sp)
                Text(
                    "Arbitrary text needs the converter-side dynamic-LSTM fix.",
                    fontSize = 12.sp,
                )
            }
        }

        Button(
            onClick = { viewModel.synthesize() },
            enabled = uiState.status != Status.RUNNING,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (uiState.status == Status.RUNNING) "Synthesizing…" else "Synthesize & Play")
        }

        when (uiState.status) {
            Status.RUNNING -> LinearProgressIndicator(Modifier.fillMaxWidth())
            Status.DONE -> Metrics(uiState)
            Status.ERROR -> Text(
                "Error: ${uiState.error}",
                color = MaterialTheme.colorScheme.error,
                fontSize = 13.sp,
            )
            else -> {}
        }
    }
}

@Composable
private fun Metrics(uiState: UiState) {
    ElevatedCard {
        Column(
            Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            MetricRow("Inference (neural)", "${uiState.inferenceMs} ms")
            MetricRow("iSTFT (host)", "${uiState.istftMs} ms")
            MetricRow("Audio", String.format("%.2f s", uiState.audioSeconds))
            MetricRow("RTF", String.format("%.3f", uiState.rtf))
            MetricRow(
                "Speed",
                String.format("%.1f× realtime", if (uiState.rtf > 0f) 1f / uiState.rtf else 0f),
            )
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, fontSize = 14.sp)
        Text(value, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace)
    }
}
