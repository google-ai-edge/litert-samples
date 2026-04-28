package com.example.gemma_on_device

import android.content.Context
import android.content.SharedPreferences
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Utility class for downloading LiteRT-LM models.
 *
 * Supports two workflows:
 *   1. In-app download from HuggingFace (with optional HF token).
 *   2. Local push via ADB:
 *        adb push gemma-4-E2B-it.litertlm /sdcard/Download/
 *      The app detects the file in the Download folder and copies it to internal storage.
 */
class ModelDownloader(private val context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ModelDownloader"
        private const val KEY_HF_TOKEN = "hf_token"

        // Gemma 4 models - NPU model first for S25 Ultra
        val AVAILABLE_MODELS = listOf(
            ModelConfig(
                id = "gemma3-1b",
                name = "Gemma 3 1B (Int4)",
                filename = "gemma3-1b-it-int4.litertlm",
                url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.litertlm",
                systemPrompt = "You are Gemma, a helpful AI assistant running on device.",
                preferredBackend = "NPU"
            ),
            ModelConfig(
                id = "gemma-3n",
                name = "Gemma 3n (Int4)",
                filename = "gemma3n.litertlm",
                url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm",
                systemPrompt = "You are Gemma, a helpful AI assistant powered by Google's LiteRT-LM running on device.",
                preferredBackend = "NPU"
            ),
            ModelConfig(
                id = "gemma4-e2b-sm8750",
                name = "Gemma 4 2B (S25 Ultra NPU)",
                filename = "gemma4_2b_181450_244_sm8750.litertlm",
                url = "local", // Pushed via ADB
                systemPrompt = "You are Gemma 4, a powerful multimodal AI assistant by Google, optimized for the Snapdragon 8 Elite NPU. You can understand text, images, and audio.",
                preferredBackend = "NPU"
            ),
            ModelConfig(
                id = "gemma4-e2b",
                name = "Gemma 4 E2B (Int4)",
                filename = "gemma-4-E2B-it.litertlm",
                url = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
                systemPrompt = "You are Gemma 4, a powerful multimodal AI assistant by Google, running privately on-device. You can understand text, images, and audio.",
                preferredBackend = "NPU"
            ),
            ModelConfig(
                id = "qwen3-0.6b",
                name = "Qwen 3 0.6B (Int4)",
                filename = "qwen3-0.6b.litertlm",
                url = "https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/qwen3-0.6b-int4.litertlm",
                systemPrompt = "You are Qwen, a helpful AI assistant running on device."
            )
        )
    }

    /**
     * Well-known directories to check for ADB-pushed model files.
     * On Android 11+, /sdcard/Download is often restricted.
     * The safest place is /sdcard/Android/data/<package>/files/
     */
    private fun getAdbSearchDirs(): List<File> {
        val dirs = mutableListOf<File>()
        
        // App-specific external storage (No permissions required)
        context.getExternalFilesDir(null)?.let { dirs.add(it) }
        context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)?.let { dirs.add(it) }
        
        // Legacy public directories (may require MANAGE_EXTERNAL_STORAGE on Android 11+)
        dirs.add(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS))
        dirs.add(File(Environment.getExternalStorageDirectory(), "Models"))
        
        return dirs
    }

    fun saveToken(token: String) {
        prefs.edit().putString(KEY_HF_TOKEN, token).apply()
    }

    fun getToken(): String? {
        return prefs.getString(KEY_HF_TOKEN, null)
    }

    /**
     * Check if a model is already available — either in internal storage
     * or ADB-pushed to a well-known external directory.
     */
    fun isModelDownloaded(modelConfig: ModelConfig): Boolean {
        // Check internal storage first
        if (File(context.filesDir, modelConfig.filename).exists()) return true
        // Check ADB push locations
        return findExternalModel(modelConfig) != null
    }

    /**
     * Search well-known external directories for an ADB-pushed model file.
     */
    private fun findExternalModel(modelConfig: ModelConfig): File? {
        for (dir in getAdbSearchDirs()) {
            val candidate = File(dir, modelConfig.filename)
            if (candidate.exists() && candidate.canRead() && candidate.length() > 0) {
                Log.i(TAG, "Found ADB-pushed model at: ${candidate.absolutePath}")
                return candidate
            }
        }
        return null
    }

    /**
     * Get the local model path. Prefers internal storage; falls back to
     * external ADB-pushed location.
     */
    fun getModelPath(modelConfig: ModelConfig): String {
        val internal = File(context.filesDir, modelConfig.filename)
        if (internal.exists()) return internal.absolutePath

        val external = findExternalModel(modelConfig)
        if (external != null) return external.absolutePath

        // Default to internal path (for download target)
        return internal.absolutePath
    }
    
    /**
     * Download model with progress reporting.
     * If the model is found in an ADB-push location, it copies from there
     * instead of downloading from HuggingFace.
     */
    fun downloadModel(modelConfig: ModelConfig): Flow<DownloadProgress> = flow {
        try {
            emit(DownloadProgress.Started)
            
            val modelFile = File(context.filesDir, modelConfig.filename)
            
            // Already in internal storage
            if (modelFile.exists()) {
                Log.d(TAG, "Model already exists at ${modelFile.absolutePath}")
                emit(DownloadProgress.Complete(modelFile.absolutePath))
                return@flow
            }

            // Check for ADB-pushed file and use it directly (no copy needed)
            val externalFile = findExternalModel(modelConfig)
            if (externalFile != null) {
                Log.i(TAG, "Using ADB-pushed model directly from: ${externalFile.absolutePath}")
                emit(DownloadProgress.Complete(externalFile.absolutePath))
                return@flow
            }
            
            // Download from HuggingFace
            Log.d(TAG, "Downloading model from ${modelConfig.url}")
            
            val url = URL(modelConfig.url)
            val connection = url.openConnection() as HttpURLConnection
            
            // Add Authorization header if token exists
            val token = getToken()
            if (!token.isNullOrBlank()) {
                Log.d(TAG, "Using HF Token for authentication")
                connection.setRequestProperty("Authorization", "Bearer $token")
            }
            
            connection.connect()
            
            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("Server returned HTTP $responseCode: ${connection.responseMessage}")
            }
            
            val fileLength = connection.contentLength
            
            connection.inputStream.use { input ->
                FileOutputStream(modelFile).use { output ->
                    val buffer = ByteArray(8192)
                    var total: Long = 0
                    var count: Int
                    
                    while (input.read(buffer).also { count = it } != -1) {
                        total += count
                        output.write(buffer, 0, count)
                        
                        if (fileLength > 0) {
                            val progress = (total * 100 / fileLength).toInt()
                            emit(DownloadProgress.Progress(progress, total, fileLength.toLong()))
                        }
                    }
                }
            }
            
            Log.d(TAG, "Model downloaded successfully to ${modelFile.absolutePath}")
            emit(DownloadProgress.Complete(modelFile.absolutePath))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
}

/**
 * Sealed class representing download progress states
 */
sealed class DownloadProgress {
    object Started : DownloadProgress()
    data class Progress(val percentage: Int, val downloaded: Long, val total: Long) : DownloadProgress()
    data class Complete(val filePath: String) : DownloadProgress()
    data class Error(val message: String) : DownloadProgress()
}
