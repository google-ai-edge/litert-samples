package com.example.qnn_litertlm_gemma

import android.content.Context
import java.io.File

/**
 * Local model resolver for the supported NPU LiteRT-LM models.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        val GEMMA4_NPU = ModelConfig(
            id = "gemma4-e2b-sm8750",
            name = "Gemma 4 2B (S25 Ultra NPU)",
            filename = "model.litertlm",
            systemPrompt = "You are Gemma 4, a powerful multimodal AI assistant by Google, optimized for the Snapdragon 8 Elite NPU. You can understand text, images, and audio.",
            preferredBackend = "NPU",
            supportsImage = true,
            supportsAudio = true,
            defaultPrompt = "Describe what you see/hear."
        )

        val FASTVLM_NPU = ModelConfig(
            id = "fastvlm-0.5b-sm8750",
            name = "FastVLM 0.5B (S25 Ultra NPU)",
            filename = "FastVLM-0.5B.qualcomm.sm8750.litertlm",
            systemPrompt = "You are FastVLM, a compact vision-language assistant optimized for the Snapdragon 8 Elite NPU. Answer concisely about the provided image.",
            preferredBackend = "NPU",
            supportsImage = true,
            supportsAudio = false,
            defaultPrompt = "Describe this image."
        )

        val AVAILABLE_MODELS = listOf(GEMMA4_NPU, FASTVLM_NPU)
    }

    fun getModelFile(modelConfig: ModelConfig = GEMMA4_NPU): File {
        val modelDir = context.getExternalFilesDir(null)
            ?: File("/sdcard/Android/data/${context.packageName}/files")
        return File(modelDir, modelConfig.filename)
    }

    fun isModelAvailable(modelConfig: ModelConfig = GEMMA4_NPU): Boolean {
        val file = getModelFile(modelConfig)
        return file.exists() && file.canRead() && file.length() > 0
    }

    fun getModelPath(modelConfig: ModelConfig = GEMMA4_NPU): String = getModelFile(modelConfig).absolutePath
}
