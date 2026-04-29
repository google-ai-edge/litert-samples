package com.example.qnn_litertlm_gemma

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
