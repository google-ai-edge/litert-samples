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

package com.google.ai.edge.examples.litert_model_zoo.view

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.google.ai.edge.examples.litert_model_zoo.MainViewModel
import com.google.ai.edge.examples.litert_model_zoo.R
import com.google.ai.edge.examples.litert_model_zoo.UiState
import java.util.Locale

/** An invented sentence and the four options of the promise / request / plan question. */
private object TextDecisionDefaults {
  const val TEXT = "I'll bring the folding chairs over tomorrow morning."
  const val QUESTION = "What is this sentence?"
  val OPTIONS =
    listOf(
      "nothing: an opinion, a story, a vague maybe, or something happening right now",
      "promise: the speaker commits to do something later",
      "request: the speaker asks the listener to do something",
      "plan: a time or day agreed to meet or do something",
    )
}

@Composable
internal fun TextDecisionPanel(state: UiState, vm: MainViewModel) {
  var text by rememberSaveable { mutableStateOf(TextDecisionDefaults.TEXT) }
  var question by rememberSaveable { mutableStateOf(TextDecisionDefaults.QUESTION) }
  var options by
    rememberSaveable(
      saver = listSaver(save = { it.value }, restore = { mutableStateOf(it.toList()) })
    ) {
      mutableStateOf(TextDecisionDefaults.OPTIONS)
    }
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    Text(
      "Ask one question about a text. Write each option as key: description; the model scores " +
        "every option and the result shows their probabilities.",
      style = MaterialTheme.typography.bodyMedium,
    )
    OutlinedTextField(
      value = text,
      onValueChange = { text = it },
      modifier = Modifier.fillMaxWidth(),
      label = { Text("Text") },
      enabled = !state.busy,
      minLines = 2,
      maxLines = 5,
    )
    OutlinedTextField(
      value = question,
      onValueChange = { question = it },
      modifier = Modifier.fillMaxWidth(),
      label = { Text("Question") },
      enabled = !state.busy,
      singleLine = true,
    )
    options.forEachIndexed { index, option ->
      OutlinedTextField(
        value = option,
        onValueChange = { value ->
          options = options.toMutableList().also { it[index] = value }
        },
        modifier = Modifier.fillMaxWidth(),
        label = { Text("Option ${index + 1} (key: description)") },
        enabled = !state.busy,
        singleLine = true,
      )
    }
    Button(
      onClick = { vm.decideText(text, question, options) },
      enabled =
        !state.busy &&
          text.isNotBlank() &&
          question.isNotBlank() &&
          options.count { it.isNotBlank() } >= 2,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Text(stringResource(R.string.run_image))
    }
  }
}

/** One bar per option in request order; the chosen key is bold. */
@Composable
internal fun TextDecisionResultPanel(state: UiState) {
  val result = state.textDecision ?: return
  Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
    Text("Answer: ${result.answerKey}", style = MaterialTheme.typography.titleMedium)
    result.probabilities.forEach { (key, probability) ->
      val chosen = key == result.answerKey
      Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
      ) {
        Text(
          key,
          Modifier.width(88.dp),
          style =
            MaterialTheme.typography.bodyMedium.copy(
              fontWeight = if (chosen) FontWeight.Bold else FontWeight.Normal
            ),
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
        )
        LinearProgressIndicator(
          progress = { probability.coerceIn(0f, 1f) },
          modifier = Modifier.weight(1f),
        )
        Text(
          String.format(Locale.ENGLISH, "%.1f%%", probability * 100),
          Modifier.width(56.dp),
          style = MaterialTheme.typography.bodySmall,
        )
      }
    }
    Text(
      "Window ${result.window} tokens",
      style = MaterialTheme.typography.bodySmall,
      color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
  }
}
