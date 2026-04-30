package com.example.gemma3_on_device

/**
 * Data class representing a LiteRT-LM model configuration
 */
data class ModelConfig(
    val id: String,
    val name: String,
    val filename: String,
    val url: String,
    val systemPrompt: String? = null,
    val preferredBackend: String? = null
)
