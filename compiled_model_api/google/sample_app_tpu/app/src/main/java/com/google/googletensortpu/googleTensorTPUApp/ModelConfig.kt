package com.google.googletensortpu.googleTensorTPUApp

/**
 * Data class representing a LiteRT-LM model configuration
 */
data class ModelConfig(
    val id: String,
    val name: String,
    val filename: String,
    val isAsset: Boolean,
    val systemPrompt: String? = null,
    val preferredBackend: String? = null,
    val supportsImage: Boolean = true,
    val supportsAudio: Boolean = false,
    val defaultPrompt: String = "Describe what you see.",
    val maxContext: Int = 2048
)
