package com.example.gemma3_on_device

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
 * Gemma 3 Edition.
 */
class ModelDownloader(private val context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ModelDownloader"
        private const val KEY_HF_TOKEN = "hf_token"

        // Models
        val AVAILABLE_MODELS = listOf(
            ModelConfig(
                id = "gemma3-1b",
                name = "Gemma 3 1B (Int4)",
                filename = "gemma3-1b-it-int4.litertlm",
                url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.litertlm",
                systemPrompt = "You are Gemma 3, a powerful multimodal AI assistant by Google, running privately on-device. You can understand text, images, and audio.",
                preferredBackend = "GPU"
            )
        )
    }

    private fun getAdbSearchDirs(): List<File> {
        val dirs = mutableListOf<File>()
        context.getExternalFilesDir(null)?.let { dirs.add(it) }
        context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)?.let { dirs.add(it) }
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

    fun isModelDownloaded(modelConfig: ModelConfig): Boolean {
        if (File(context.filesDir, modelConfig.filename).exists()) return true
        return findExternalModel(modelConfig) != null
    }

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

    fun getModelPath(modelConfig: ModelConfig): String {
        val internal = File(context.filesDir, modelConfig.filename)
        if (internal.exists()) return internal.absolutePath
        val external = findExternalModel(modelConfig)
        if (external != null) return external.absolutePath
        return internal.absolutePath
    }
    
    fun downloadModel(modelConfig: ModelConfig): Flow<DownloadProgress> = flow {
        try {
            emit(DownloadProgress.Started)
            val modelFile = File(context.filesDir, modelConfig.filename)
            if (modelFile.exists()) {
                emit(DownloadProgress.Complete(modelFile.absolutePath))
                return@flow
            }
            val externalFile = findExternalModel(modelConfig)
            if (externalFile != null) {
                emit(DownloadProgress.Complete(externalFile.absolutePath))
                return@flow
            }
            
            val url = URL(modelConfig.url)
            val connection = url.openConnection() as HttpURLConnection
            val token = getToken()
            if (!token.isNullOrBlank()) {
                connection.setRequestProperty("Authorization", "Bearer $token")
            }
            connection.connect()
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("Server returned HTTP ${connection.responseCode}")
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
            emit(DownloadProgress.Complete(modelFile.absolutePath))
        } catch (e: Exception) {
            emit(DownloadProgress.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
}

sealed class DownloadProgress {
    object Started : DownloadProgress()
    data class Progress(val percentage: Int, val downloaded: Long, val total: Long) : DownloadProgress()
    data class Complete(val filePath: String) : DownloadProgress()
    data class Error(val message: String) : DownloadProgress()
}
