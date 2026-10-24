package com.example.gemma3_npu

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
 * Resolves locally pushed or downloaded LiteRT-LM packages for Gemma 3 1B on Qualcomm NPU.
 */
class ModelDownloader(private val context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val TAG = "ModelDownloader"
        private const val KEY_HF_TOKEN = "hf_token"

        val AVAILABLE_MODELS = listOf(
            ModelConfig(
                id = "gemma3-npu",
                name = "Gemma 3 1B (NPU-optimized)",
                filename = "Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm",
                url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm",
                systemPrompt = "You are Gemma 3, a helpful AI assistant by Google running privately on-device.",
                preferredBackend = "NPU"
            )
        )

        internal fun findReadableModel(filename: String, searchDirs: List<File>): File? {
            for (dir in searchDirs) {
                val candidate = File(dir, filename)
                if (candidate.isFile && candidate.canRead() && candidate.length() > 0L) {
                    return candidate
                }
            }
            return null
        }
    }

    private fun getSearchDirs(): List<File> {
        val dirs = mutableListOf<File>()
        dirs.add(context.filesDir)
        context.getExternalFilesDir(null)?.let { dirs.add(it) }
        dirs.add(File("/sdcard/Android/data/${context.packageName}/files"))
        context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)?.let { dirs.add(it) }
        dirs.add(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS))
        dirs.add(File(Environment.getExternalStorageDirectory(), "Models"))
        return dirs.distinctBy { it.absolutePath }
    }

    fun saveToken(token: String) {
        prefs.edit().putString(KEY_HF_TOKEN, token).apply()
    }

    fun getToken(): String? = prefs.getString(KEY_HF_TOKEN, null)

    fun isModelDownloaded(modelConfig: ModelConfig): Boolean =
        findReadableModel(modelConfig.filename, getSearchDirs()) != null

    fun getModelPath(modelConfig: ModelConfig): String {
        findReadableModel(modelConfig.filename, getSearchDirs())?.let {
            Log.i(TAG, "Resolved model path: ${it.absolutePath}")
            return it.absolutePath
        }
        return File(context.filesDir, modelConfig.filename).absolutePath
    }

    fun getAdbPushCommand(modelConfig: ModelConfig): String =
        "adb push ${modelConfig.filename} /sdcard/Android/data/${context.packageName}/files/"

    fun downloadModel(modelConfig: ModelConfig): Flow<DownloadProgress> = flow {
        val targetFile = File(context.filesDir, modelConfig.filename)
        val tempFile = File(context.filesDir, "${modelConfig.filename}.part")
        try {
            emit(DownloadProgress.Started)
            findReadableModel(modelConfig.filename, getSearchDirs())?.let { existing ->
                emit(DownloadProgress.Complete(existing.absolutePath))
                return@flow
            }

            val connection = (URL(modelConfig.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 60_000
                val token = getToken()
                if (!token.isNullOrBlank()) {
                    setRequestProperty("Authorization", "Bearer $token")
                }
            }

            connection.connect()
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("Server returned HTTP ${connection.responseCode}")
            }

            val fileLength = connection.contentLengthLong
            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var total = 0L
                    var count: Int
                    while (input.read(buffer).also { count = it } != -1) {
                        total += count
                        output.write(buffer, 0, count)
                        if (fileLength > 0L) {
                            val progress = ((total * 100L) / fileLength).toInt()
                            emit(DownloadProgress.Progress(progress, total, fileLength))
                        }
                    }
                }
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }
            if (!tempFile.renameTo(targetFile)) {
                tempFile.copyTo(targetFile, overwrite = true)
                tempFile.delete()
            }
            emit(DownloadProgress.Complete(targetFile.absolutePath))
        } catch (e: Exception) {
            tempFile.delete()
            emit(DownloadProgress.Error(e.message ?: "Unknown download error"))
        }
    }.flowOn(Dispatchers.IO)
}

sealed class DownloadProgress {
    object Started : DownloadProgress()
    data class Progress(val percentage: Int, val downloaded: Long, val total: Long) : DownloadProgress()
    data class Complete(val filePath: String) : DownloadProgress()
    data class Error(val message: String) : DownloadProgress()
}
