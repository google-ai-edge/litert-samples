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

package com.google.aiedge.examples.phototalk

import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val sender: Sender,
    val text: String,
    val isStreaming: Boolean = false
) {
    enum class Sender { USER, AI, SYSTEM }
}

sealed class ClassificationUiState {
    object Idle : ClassificationUiState()
    object Classifying : ClassificationUiState()
    data class Success(val result: ClassificationResult) : ClassificationUiState()
    data class Error(val message: String) : ClassificationUiState()
}

data class PhotoTalkUiState(
    val selectedImageUri: Uri? = null,
    val selectedBitmap: Bitmap? = null,
    val classificationState: ClassificationUiState = ClassificationUiState.Idle,
    val modelPath: String = "/sdcard/Download/gemma-2b-it.litertlm",
    val isLmEngineReady: Boolean = false,
    val isLmInitializing: Boolean = false,
    val lmBackendName: String = "CPU",
    val statusMessage: String = "Select an image to start classification and chat.",
    val chatMessages: List<ChatMessage> = emptyList(),
    val isStreamingResponse: Boolean = false
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(PhotoTalkUiState())
    val uiState: StateFlow<PhotoTalkUiState> = _uiState.asStateFlow()

    private val classifierHelper = ImageClassifierHelper(application)
    private val lmHelper = LiteRtLmHelper.getInstance(application)

    companion object {
        private const val TAG = "PhotoTalk_VM"
    }

    fun updateModelPath(newPath: String) {
        _uiState.update { it.copy(modelPath = newPath) }
    }

    fun initializeLmEngine() {
        viewModelScope.launch {
            ensureLmEngineInitialized()
        }
    }

    private suspend fun ensureLmEngineInitialized(): Boolean {
        if (_uiState.value.isLmEngineReady) return true
        val path = _uiState.value.modelPath
        if (path.isBlank()) {
            _uiState.update { it.copy(statusMessage = "Please set a valid .litertlm model path in Settings.") }
            return false
        }

        _uiState.update { it.copy(isLmInitializing = true, statusMessage = "Initializing LiteRT-LM Engine...") }
        val result = lmHelper.initializeEngine(path)
        return result.fold(
            onSuccess = {
                _uiState.update {
                    it.copy(
                        isLmEngineReady = true,
                        isLmInitializing = false,
                        lmBackendName = lmHelper.getActiveBackend(),
                        statusMessage = "LiteRT-LM Ready (${lmHelper.getActiveBackend()}). Select an image!"
                    )
                }
                true
            },
            onFailure = { err ->
                _uiState.update {
                    it.copy(
                        isLmEngineReady = false,
                        isLmInitializing = false,
                        statusMessage = "LiteRT-LM Init Info: ${err.localizedMessage}"
                    )
                }
                false
            }
        )
    }

    fun onImageSelected(uri: Uri) {
        val context = getApplication<Application>()
        try {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri))
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            }.copy(Bitmap.Config.ARGB_8888, true)

            _uiState.update {
                it.copy(
                    selectedImageUri = uri,
                    selectedBitmap = bitmap,
                    classificationState = ClassificationUiState.Classifying,
                    statusMessage = "Running LiteRT CompiledModel classification..."
                )
            }

            processImage(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load image bitmap", e)
            _uiState.update {
                it.copy(
                    classificationState = ClassificationUiState.Error("Failed to decode image"),
                    statusMessage = "Error loading image: ${e.localizedMessage}"
                )
            }
        }
    }

    private fun processImage(bitmap: Bitmap) {
        viewModelScope.launch {
            val classification = classifierHelper.classify(bitmap)
            if (classification != null) {
                val pct = classification.confidence * 100f
                _uiState.update {
                    it.copy(
                        classificationState = ClassificationUiState.Success(classification),
                        statusMessage = "LiteRT Detected: ${classification.label} (${"%.1f".format(pct)}%)."
                    )
                }
                startCoDependentConversation(classification.label, pct)
            } else {
                _uiState.update {
                    it.copy(
                        classificationState = ClassificationUiState.Error("LiteRT classification returned no result."),
                        statusMessage = "Classification failed."
                    )
                }
            }
        }
    }

    private suspend fun startCoDependentConversation(label: String, confidencePct: Float) {
        val lmReady = ensureLmEngineInitialized()
        if (!lmReady) {
            val systemMsg = ChatMessage(
                sender = ChatMessage.Sender.SYSTEM,
                text = "⚡ LiteRT classified image as '$label' (${"%.1f".format(confidencePct)}%). " +
                        "Note: LiteRT-LM model not initialized. Download a .litertlm file and update the path in Settings (⚙️) to start AI Chat."
            )
            _uiState.update { it.copy(chatMessages = listOf(systemMsg)) }
            return
        }

        val initialSystemMsg = ChatMessage(
            sender = ChatMessage.Sender.SYSTEM,
            text = "⚡ Pipeline: LiteRT classified image as '$label' (${"%.1f".format(confidencePct)}%). Handoff to LiteRT-LM..."
        )
        _uiState.update { it.copy(chatMessages = listOf(initialSystemMsg), isStreamingResponse = true) }

        val aiMessageId = java.util.UUID.randomUUID().toString()
        val streamingAiMsg = ChatMessage(id = aiMessageId, sender = ChatMessage.Sender.AI, text = "", isStreaming = true)
        _uiState.update { it.copy(chatMessages = it.chatMessages + streamingAiMsg) }

        try {
            var accumulatedText = ""
            lmHelper.startImageConversation(label, confidencePct)
                .catch { err ->
                    Log.e(TAG, "LLM streaming error", err)
                    updateChatMessage(aiMessageId, "Error generating AI response: ${err.localizedMessage}", isStreaming = false)
                    _uiState.update { it.copy(isStreamingResponse = false) }
                }
                .collect { chunk ->
                    accumulatedText += chunk
                    updateChatMessage(aiMessageId, accumulatedText, isStreaming = true)
                }

            updateChatMessage(aiMessageId, accumulatedText, isStreaming = false)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start image conversation", e)
            updateChatMessage(aiMessageId, "Failed to start conversation: ${e.localizedMessage}", isStreaming = false)
        } finally {
            _uiState.update { it.copy(isStreamingResponse = false, statusMessage = "Chat active for $label") }
        }
    }

    fun sendMessage(userText: String) {
        if (userText.isBlank() || _uiState.value.isStreamingResponse) return

        val userMsg = ChatMessage(sender = ChatMessage.Sender.USER, text = userText)
        val aiMessageId = java.util.UUID.randomUUID().toString()
        val streamingAiMsg = ChatMessage(id = aiMessageId, sender = ChatMessage.Sender.AI, text = "", isStreaming = true)

        _uiState.update {
            it.copy(
                chatMessages = it.chatMessages + userMsg + streamingAiMsg,
                isStreamingResponse = true
            )
        }

        viewModelScope.launch {
            var accumulatedText = ""
            lmHelper.sendChatMessage(userText)
                .catch { err ->
                    Log.e(TAG, "Chat streaming error", err)
                    updateChatMessage(aiMessageId, "Error: ${err.localizedMessage}", isStreaming = false)
                    _uiState.update { it.copy(isStreamingResponse = false) }
                }
                .collect { chunk ->
                    accumulatedText += chunk
                    updateChatMessage(aiMessageId, accumulatedText, isStreaming = true)
                }

            updateChatMessage(aiMessageId, accumulatedText, isStreaming = false)
            _uiState.update { it.copy(isStreamingResponse = false) }
        }
    }

    private fun updateChatMessage(id: String, text: String, isStreaming: Boolean) {
        _uiState.update { state ->
            val updatedList = state.chatMessages.map { msg ->
                if (msg.id == id) msg.copy(text = text, isStreaming = isStreaming) else msg
            }
            state.copy(chatMessages = updatedList)
        }
    }

    override fun onCleared() {
        super.onCleared()
        classifierHelper.close()
        lmHelper.cleanup()
    }
}
