package com.google.googletensortpu.googleTensorTPUApp

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

class ModelResolver(private val context: Context) {

    companion object {
        private const val TAG = "ModelResolver"

        val GEMMA3_CPU = ModelConfig(
            id = "gemma3-cpu",
            name = "Gemma 3 (Assets - CPU)",
            filename = "gemma3_cpu.litertlm",
            isAsset = true,
            systemPrompt = "You are a helpful assistant.",
            preferredBackend = "CPU",
            supportsImage = false,
            supportsAudio = false,
            defaultPrompt = "What is this?",
            maxContext = 2048
        )

        val TINY_GARDEN = ModelConfig(
            id = "tiny-garden",
            name = "Tiny Garden (Assets - CPU)",
            filename = "tiny_garden.litertlm",
            isAsset = true,
            systemPrompt = "You are a helpful assistant.",
            preferredBackend = "CPU",
            supportsImage = false,
            supportsAudio = false,
            defaultPrompt = "Analyze the image.",
            maxContext = 2048
        )

        val AVAILABLE_MODELS = mutableListOf(GEMMA3_CPU, TINY_GARDEN)

        private const val PREFS_NAME = "ModelResolverPrefs"
        private const val KEY_CUSTOM_MODELS = "CustomModels"

        fun loadCustomModels(context: Context) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val json = prefs.getString(KEY_CUSTOM_MODELS, null)
            if (json != null) {
                try {
                    val array = org.json.JSONArray(json)
                    for (i in 0 until array.length()) {
                        val obj = array.getJSONObject(i)
                        val config = ModelConfig(
                            id = obj.getString("id"),
                            name = obj.getString("name"),
                            filename = obj.getString("filename"),
                            isAsset = obj.getBoolean("isAsset"),
                            systemPrompt = if (obj.has("systemPrompt")) obj.getString("systemPrompt") else null,
                            preferredBackend = if (obj.has("preferredBackend")) obj.getString("preferredBackend") else null,
                            supportsImage = obj.getBoolean("supportsImage"),
                            supportsAudio = obj.getBoolean("supportsAudio"),
                            defaultPrompt = obj.getString("defaultPrompt"),
                            maxContext = if (obj.has("maxContext")) obj.getInt("maxContext") else 2048
                        )
                        if (AVAILABLE_MODELS.none { it.id == config.id }) {
                            AVAILABLE_MODELS.add(config)
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }

        fun saveCustomModel(context: Context, config: ModelConfig) {
            if (AVAILABLE_MODELS.none { it.id == config.id }) {
                AVAILABLE_MODELS.add(config)
            }
            
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val customModels = AVAILABLE_MODELS.filter { !it.isAsset }
            val array = org.json.JSONArray()
            for (model in customModels) {
                val obj = org.json.JSONObject().apply {
                    put("id", model.id)
                    put("name", model.name)
                    put("filename", model.filename)
                    put("isAsset", model.isAsset)
                    put("systemPrompt", model.systemPrompt ?: "")
                    put("preferredBackend", model.preferredBackend ?: "")
                    put("supportsImage", model.supportsImage)
                    put("supportsAudio", model.supportsAudio)
                    put("defaultPrompt", model.defaultPrompt)
                    put("maxContext", model.maxContext)
                }
                array.put(obj)
            }
            prefs.edit().putString(KEY_CUSTOM_MODELS, array.toString()).apply()
        }

        fun deleteCustomModel(context: Context, config: ModelConfig) {
            AVAILABLE_MODELS.removeAll { it.id == config.id }
            
            // Remove from preferences by re-saving
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val customModels = AVAILABLE_MODELS.filter { !it.isAsset }
            val array = org.json.JSONArray()
            for (model in customModels) {
                val obj = org.json.JSONObject().apply {
                    put("id", model.id)
                    put("name", model.name)
                    put("filename", model.filename)
                    put("isAsset", model.isAsset)
                    put("systemPrompt", model.systemPrompt ?: "")
                    put("preferredBackend", model.preferredBackend ?: "")
                    put("supportsImage", model.supportsImage)
                    put("supportsAudio", model.supportsAudio)
                    put("defaultPrompt", model.defaultPrompt)
                }
                array.put(obj)
            }
            prefs.edit().putString(KEY_CUSTOM_MODELS, array.toString()).apply()

            // Delete file
            if (!config.isAsset) {
                val file = java.io.File(context.getExternalFilesDir(null), config.filename)
                if (file.exists()) {
                    file.delete()
                }
            }
        }
    }

    /**
     * Resolves the final local absolute path for the model.
     * For assets, it copies the file to context.cacheDir if it does not exist.
     * For external models, it references the external files directory.
     */
    fun resolveModelPath(modelConfig: ModelConfig): String {
        if (modelConfig.isAsset) {
            val cacheFile = File(context.cacheDir, modelConfig.filename)
            if (!cacheFile.exists() || cacheFile.length() == 0L) {
                try {
                    Log.i(TAG, "Copying asset ${modelConfig.filename} to cache...")
                    context.assets.open(modelConfig.filename).use { input ->
                        FileOutputStream(cacheFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.i(TAG, "Asset copy complete: ${cacheFile.length()} bytes")
                } catch (e: Exception) {
                    Log.e(TAG, "Asset ${modelConfig.filename} not found or copy failed", e)
                }
            }
            return cacheFile.absolutePath
        } else {
            // Prioritize internal cache directory to bypass SELinux boundaries for TPU/NPU HAL
            val cacheFile = File(context.cacheDir, modelConfig.filename)
            val externalDir = context.getExternalFilesDir(null)
                ?: File("/sdcard/Android/data/${context.packageName}/files")
            val externalFile = File(externalDir, modelConfig.filename)
            
            if (cacheFile.exists() && cacheFile.length() == externalFile.length()) {
                Log.d(TAG, "Using model from internal cache: ${cacheFile.absolutePath}")
                return cacheFile.absolutePath
            }
            
            if (externalFile.exists()) {
                try {
                    Log.i(TAG, "Copying external model ${modelConfig.filename} to cache to bypass SELinux...")
                    externalFile.inputStream().use { input ->
                        FileOutputStream(cacheFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.i(TAG, "External model copy complete: ${cacheFile.length()} bytes")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to copy external model ${modelConfig.filename}", e)
                }
            }
            
            return cacheFile.absolutePath
        }
    }

    /**
     * Checks if the model is ready to be loaded.
     * Assets are always checked against context.assets or cacheDir.
     * External files must exist, be readable, and have non-zero length.
     */
    fun isModelAvailable(modelConfig: ModelConfig): Boolean {
        val cacheFile = File(context.cacheDir, modelConfig.filename)
        if (cacheFile.exists() && cacheFile.canRead() && cacheFile.length() > 0) {
            return true
        }
        if (modelConfig.isAsset) {
            return true
        } else {
            val externalDir = context.getExternalFilesDir(null)
                ?: File("/sdcard/Android/data/${context.packageName}/files")
            val modelFile = File(externalDir, modelConfig.filename)
            val available = modelFile.exists() && modelFile.canRead() && modelFile.length() > 0
            Log.d(TAG, "External model ${modelConfig.filename} availability check: $available at ${modelFile.absolutePath}")
            return available
        }
    }
}
