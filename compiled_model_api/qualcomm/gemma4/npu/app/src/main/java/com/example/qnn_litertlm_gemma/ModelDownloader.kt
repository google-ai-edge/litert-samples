package com.example.qnn_litertlm_gemma

import android.content.Context
import java.io.File

/**
 * Local model resolver for the single supported Gemma 4 NPU model.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        val GEMMA4_NPU = ModelConfig(
            id = "gemma4-e2b-sm8750",
            name = "Gemma 4 2B (S25 Ultra NPU)",
            filename = "model.litertlm",
            systemPrompt = "You are Gemma 4, a powerful multimodal AI assistant by Google, optimized for the Snapdragon 8 Elite NPU. You can understand text, images, and audio.",
            preferredBackend = "NPU"
        )
    }

    fun getModelFile(): File {
        val modelDir = context.getExternalFilesDir(null)
            ?: File("/sdcard/Android/data/${context.packageName}/files")
        return File(modelDir, GEMMA4_NPU.filename)
    }

    fun isModelAvailable(): Boolean {
        val file = getModelFile()
        return file.exists() && file.canRead() && file.length() > 0
    }

    fun getModelPath(): String = getModelFile().absolutePath
}
