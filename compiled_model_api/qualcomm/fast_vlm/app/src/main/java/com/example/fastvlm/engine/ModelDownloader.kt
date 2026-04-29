package com.example.fastvlm.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Checks for FastVLM model files pushed via ADB to /data/local/tmp/fastvlm.
 * If found, copies them to internal storage to ensure native engine access.
 */
object ModelDownloader {

    private const val TAG = "ModelDownloader"
    private const val ADB_PATH = "/data/local/tmp/fastvlm"

    private val MODEL_FILES = mapOf(
        "NPU" to "FastVLM-0.5B.sm8850.litertlm",
        "GPU" to "FastVLM-0.5B.litertlm",
        "CPU" to "FastVLM-0.5B.litertlm",
    )

    /**
     * Returns the local path for the model file.
     * Copies from ADB_PATH to internal storage if needed.
     */
    suspend fun getModelPath(context: Context, backend: String): String {
        val fileName = MODEL_FILES[backend]
            ?: throw IllegalArgumentException("Unknown backend: $backend")
        
        val internalFile = File(context.filesDir, fileName)
        val adbFile = File(ADB_PATH, fileName)

        Log.d(TAG, "Checking for model: $fileName")
        Log.d(TAG, "ADB File: ${adbFile.absolutePath} (exists=${adbFile.exists()}, size=${adbFile.length()})")
        Log.d(TAG, "Internal File: ${internalFile.absolutePath} (exists=${internalFile.exists()}, size=${internalFile.length()})")

        // 1. Check if model exists in /data/local/tmp
        if (adbFile.exists() && adbFile.length() > 0) {
            // 2. If it's not in internal storage, or if adb version is different size, copy it
            if (!internalFile.exists() || internalFile.length() != adbFile.length()) {
                Log.d(TAG, "Copying model from /data/local/tmp to internal storage... ($fileName)")
                copyFile(adbFile, internalFile)
                Log.d(TAG, "Copy complete. New internal size: ${internalFile.length()}")
            }
            return internalFile.absolutePath
        }

        // 3. Fallback: check if already in internal storage
        if (internalFile.exists() && internalFile.length() > 0) {
            return internalFile.absolutePath
        }

        throw ModelNotDownloadedException(
            "Model file not found. Please push to $ADB_PATH using adb.",
            fileName
        )
    }

    private suspend fun copyFile(source: File, destination: File) {
        withContext(Dispatchers.IO) {
            val tempFile = File(destination.parent, "${destination.name}.tmp")
            FileInputStream(source).use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }
            if (destination.exists()) destination.delete()
            tempFile.renameTo(destination)
        }
    }

    /**
     * Checks if the model is available.
     */
    fun isModelDownloaded(context: Context, backend: String): Boolean {
        val fileName = MODEL_FILES[backend] ?: return false
        val internalFile = File(context.filesDir, fileName)
        val adbFile = File(ADB_PATH, fileName)
        return (internalFile.exists() && internalFile.length() > 0) || 
               (adbFile.exists() && adbFile.length() > 0)
    }

    /**
     * No-op download.
     */
    fun downloadModel(context: Context, backend: String): Flow<Float> = flow {
        emit(1f)
    }

    fun getModelFileName(backend: String): String {
        return MODEL_FILES[backend] ?: "unknown"
    }

    fun clearCache(context: Context) {
        MODEL_FILES.values.distinct().forEach { fileName ->
            File(context.filesDir, fileName).delete()
        }
    }
}

class ModelNotDownloadedException(message: String, val fileName: String) :
    Exception(message)
