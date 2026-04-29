package com.example.fastvlm.model

import android.net.Uri
import java.util.UUID

enum class Role {
    USER, ASSISTANT, SYSTEM
}

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: Role,
    val text: String,
    val imageUri: Uri? = null,
    val timestamp: Long = System.currentTimeMillis(),
)

enum class EngineStatus {
    IDLE,
    DOWNLOADING,
    INITIALIZING,
    READY,
    GENERATING,
    ERROR
}

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val engineStatus: EngineStatus = EngineStatus.IDLE,
    val downloadProgress: Float = 0f,
    val statusMessage: String = "Tap to initialize",
    val selectedBackend: String = "NPU",
    val currentImageUri: Uri? = null,
    val errorMessage: String? = null,
)
