package com.example.qnn_litertlm_gemma

/**
 * Configuration metadata for a supported on-device LiteRT-LM model package.
 */
data class ModelConfig(
    val id: String,
    val name: String,
    val filename: String,
    val candidateFilenames: List<String> = listOf(filename),
    val systemPrompt: String? = null,
    val preferredBackend: String? = null,
    val supportsImage: Boolean = true,
    val supportsAudio: Boolean = false,
    val defaultPrompt: String = "Describe what you see."
)
