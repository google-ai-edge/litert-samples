package com.example.fastvlm.viewmodel

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.fastvlm.engine.InferenceEngine
import com.example.fastvlm.engine.ModelDownloader
import com.example.fastvlm.model.ChatMessage
import com.example.fastvlm.model.ChatUiState
import com.example.fastvlm.model.EngineStatus
import com.example.fastvlm.model.Role
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "ChatViewModel"
    }

    private val engine = InferenceEngine(application.applicationContext)

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    /**
     * Start the full initialization pipeline: download model (if needed) then init engine.
     */
    fun initializeEngine(backend: String = "NPU") {
        viewModelScope.launch {
            try {
                _uiState.update { it.copy(selectedBackend = backend, errorMessage = null) }

                // Step 1: Check if model is downloaded
                val context = getApplication<Application>().applicationContext
                if (!ModelDownloader.isModelDownloaded(context, backend)) {
                    _uiState.update {
                        it.copy(
                            engineStatus = EngineStatus.DOWNLOADING,
                            statusMessage = "Downloading ${ModelDownloader.getModelFileName(backend)}...",
                            downloadProgress = 0f
                        )
                    }

                    ModelDownloader.downloadModel(context, backend).collect { progress ->
                        _uiState.update {
                            it.copy(
                                downloadProgress = progress,
                                statusMessage = "Downloading... ${(progress * 100).toInt()}%"
                            )
                        }
                    }

                    _uiState.update {
                        it.copy(
                            statusMessage = "Download complete",
                            downloadProgress = 1f
                        )
                    }
                }

                // Step 2: Initialize the engine
                _uiState.update {
                    it.copy(
                        engineStatus = EngineStatus.INITIALIZING,
                        statusMessage = "Initializing $backend engine..."
                    )
                }

                engine.initialize(backend)
                val actualBackend = engine.getCurrentBackend()

                _uiState.update {
                    it.copy(
                        selectedBackend = actualBackend,
                        engineStatus = EngineStatus.READY,
                        statusMessage = "Ready on $actualBackend"
                    )
                }

                Log.d(TAG, "Engine ready on $actualBackend")

            } catch (e: Exception) {
                Log.e(TAG, "Initialization failed", e)
                _uiState.update {
                    it.copy(
                        engineStatus = EngineStatus.ERROR,
                        statusMessage = "Error: ${e.message}",
                        errorMessage = e.message
                    )
                }
            }
        }
    }

    /**
     * Send a message (text-only or multimodal with image).
     */
    fun sendMessage(text: String) {
        if (text.isBlank() && _uiState.value.currentImageUri == null) return
        if (_uiState.value.engineStatus != EngineStatus.READY) return

        val imageUri = _uiState.value.currentImageUri
        val displayText = text.ifBlank { "Describe this image." }

        // Add user message
        val userMessage = ChatMessage(
            role = Role.USER,
            text = displayText,
            imageUri = imageUri
        )

        _uiState.update {
            it.copy(
                messages = it.messages + userMessage,
                engineStatus = EngineStatus.GENERATING,
                statusMessage = "Generating...",
                currentImageUri = null, // Clear after sending
            )
        }

        // Add placeholder assistant message
        val assistantMessageId = java.util.UUID.randomUUID().toString()
        val assistantMessage = ChatMessage(
            id = assistantMessageId,
            role = Role.ASSISTANT,
            text = ""
        )

        _uiState.update {
            it.copy(messages = it.messages + assistantMessage)
        }

        // Start generation
        generationJob = viewModelScope.launch {
            try {
                val responseFlow = if (imageUri != null) {
                    engine.sendMultimodalMessage(displayText, imageUri)
                } else {
                    engine.sendTextMessage(displayText)
                }

                val responseBuilder = StringBuilder()

                responseFlow.collect { token ->
                    responseBuilder.append(token)
                    val currentText = responseBuilder.toString()

                    _uiState.update { state ->
                        state.copy(
                            messages = state.messages.map { msg ->
                                if (msg.id == assistantMessageId) {
                                    msg.copy(text = currentText)
                                } else msg
                            }
                        )
                    }
                }

                _uiState.update {
                    it.copy(
                        engineStatus = EngineStatus.READY,
                        statusMessage = "Ready on ${it.selectedBackend}"
                    )
                }

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)

                // Update assistant message with error
                _uiState.update { state ->
                    state.copy(
                        messages = state.messages.map { msg ->
                            if (msg.id == assistantMessageId) {
                                msg.copy(text = "[Error: ${e.message}]")
                            } else msg
                        },
                        engineStatus = EngineStatus.READY,
                        statusMessage = "Ready on ${state.selectedBackend}",
                        errorMessage = e.message
                    )
                }
            }
        }
    }

    /**
     * Set the image to be sent with the next message.
     */
    fun setImage(uri: Uri?) {
        _uiState.update { it.copy(currentImageUri = uri) }
    }

    /**
     * Clear the attached image.
     */
    fun clearImage() {
        _uiState.update { it.copy(currentImageUri = null) }
    }

    /**
     * Reset the conversation.
     */
    fun resetConversation() {
        generationJob?.cancel()
        viewModelScope.launch {
            try {
                engine.resetConversation()
                _uiState.update {
                    it.copy(
                        messages = emptyList(),
                        engineStatus = EngineStatus.READY,
                        statusMessage = "Ready on ${it.selectedBackend}",
                        currentImageUri = null,
                        errorMessage = null
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Reset failed", e)
            }
        }
    }

    /**
     * Switch to a different backend. Re-downloads model if needed.
     */
    fun switchBackend(backend: String) {
        if (backend == _uiState.value.selectedBackend && engine.isInitialized()) return
        generationJob?.cancel()
        _uiState.update {
            it.copy(
                messages = emptyList(),
                currentImageUri = null,
                errorMessage = null
            )
        }
        initializeEngine(backend)
    }

    override fun onCleared() {
        super.onCleared()
        generationJob?.cancel()
        engine.close()
    }
}
