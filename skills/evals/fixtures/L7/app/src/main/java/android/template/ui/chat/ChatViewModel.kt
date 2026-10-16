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

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ChatMessage(val fromUser: Boolean, val text: String)

data class ChatState(val messages: List<ChatMessage> = emptyList(), val busy: Boolean = false)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val modelPath = File(app.filesDir, "model.litertlm").absolutePath
    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    fun send(text: String) {
        _state.update {
            it.copy(
                messages = it.messages + ChatMessage(true, text) + ChatMessage(false, ""),
                busy = true,
            )
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val engine = Engine(EngineConfig(modelPath = modelPath, backend = Backend.CPU()))
                engine.initialize()
                val conversation = engine.createConversation()
                conversation.sendMessageAsync(text).collect { chunk ->
                    appendToReply(chunk.toString())
                }
            } catch (e: Exception) {
                appendToReply("Error: ${e.message}")
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    private fun appendToReply(text: String) {
        _state.update {
            val reply = it.messages.last()
            it.copy(messages = it.messages.dropLast(1) + reply.copy(text = reply.text + text))
        }
    }
}
