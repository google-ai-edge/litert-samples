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

package android.template.ui.chat

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ChatScreen(modifier: Modifier = Modifier, viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var input by remember { mutableStateOf("") }
    Column(modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        LazyColumn(Modifier.weight(1f)) {
            items(state.messages) { message ->
                val text = if (message.fromUser) "You: ${message.text}" else message.text
                Text(text, Modifier.padding(vertical = 4.dp))
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextField(value = input, onValueChange = { input = it }, modifier = Modifier.weight(1f))
            Button(
                onClick = {
                    viewModel.send(input)
                    input = ""
                },
                enabled = !state.busy && input.isNotBlank(),
            ) {
                Text("Send")
            }
        }
    }
}
