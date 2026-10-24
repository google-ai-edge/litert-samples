package com.example.qnn_litertlm_gemma

import android.content.Context
import java.io.File

/**
 * Resolves locally provisioned LiteRT-LM NPU model packages across scoped app directories.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        val GEMMA4_NPU = ModelConfig(
            id = "gemma4-e2b-sm8750",
            name = "Gemma 4 E2B (SM8750 NPU)",
            filename = "gemma-4-E2B-it_qualcomm_sm8750.litertlm",
            candidateFilenames = listOf(
                "gemma-4-E2B-it_qualcomm_sm8750.litertlm",
                "model.litertlm",
                "gemma-4-E2B-it.litertlm"
            ),
            systemPrompt = "You are Gemma 4, a helpful multimodal AI assistant running on-device.",
            preferredBackend = "NPU",
            supportsImage = true,
            supportsAudio = true,
            defaultPrompt = "Describe what you see/hear."
        )

        val FASTVLM_NPU = ModelConfig(
            id = "fastvlm-0.5b-sm8750",
            name = "FastVLM 0.5B (SM8750 NPU)",
            filename = "FastVLM-0.5B.qualcomm.sm8750.litertlm",
            candidateFilenames = listOf("FastVLM-0.5B.qualcomm.sm8750.litertlm"),
            systemPrompt = "You are FastVLM, a concise on-device vision-language assistant.",
            preferredBackend = "NPU",
            supportsImage = true,
            supportsAudio = false,
            defaultPrompt = "Describe this image."
        )

        val AVAILABLE_MODELS = listOf(GEMMA4_NPU, FASTVLM_NPU)

        internal fun resolveModelInDirectories(
            modelConfig: ModelConfig,
            searchDirs: List<File>
        ): File? {
            val filenames = (listOf(modelConfig.filename) + modelConfig.candidateFilenames).distinct()
            for (dir in searchDirs) {
                for (name in filenames) {
                    val candidate = File(dir, name)
                    if (candidate.isFile && candidate.canRead() && candidate.length() > 0L) {
                        return candidate
                    }
                }
            }
            return null
        }
    }

    private fun getSearchDirectories(): List<File> {
        val dirs = mutableListOf<File>()
        context.getExternalFilesDir(null)?.let { dirs.add(it) }
        dirs.add(File("/sdcard/Android/data/${context.packageName}/files"))
        dirs.add(context.filesDir)
        return dirs.distinctBy { it.absolutePath }
    }

    fun getModelFile(modelConfig: ModelConfig = GEMMA4_NPU): File {
        resolveModelInDirectories(modelConfig, getSearchDirectories())?.let { return it }
        val defaultDir = context.getExternalFilesDir(null)
            ?: File("/sdcard/Android/data/${context.packageName}/files")
        return File(defaultDir, modelConfig.filename)
    }

    fun isModelAvailable(modelConfig: ModelConfig = GEMMA4_NPU): Boolean {
        return resolveModelInDirectories(modelConfig, getSearchDirectories()) != null
    }

    fun getModelPath(modelConfig: ModelConfig = GEMMA4_NPU): String =
        getModelFile(modelConfig).absolutePath

    fun getAdbPushCommand(modelConfig: ModelConfig = GEMMA4_NPU): String =
        "adb push ${modelConfig.filename} /sdcard/Android/data/${context.packageName}/files/"
}
