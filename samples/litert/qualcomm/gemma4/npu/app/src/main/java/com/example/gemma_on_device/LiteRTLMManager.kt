package com.example.gemma_on_device

import android.content.Context
import android.system.Os
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Singleton manager for LiteRT-LM Engine (NPU / GPU / CPU multimodal backends).
 */
class LiteRTLMManager private constructor(private val context: Context) {

    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"
    private var currentSystemPrompt: String? = null

    companion object {
        private const val TAG = "LiteRTLMManager"

        @Volatile
        private var nativeLoaded = false

        private fun ensureNativeLoaded() {
            if (nativeLoaded) return
            synchronized(this) {
                if (nativeLoaded) return
                try {
                    System.loadLibrary("LiteRt")
                    Log.i(TAG, "Loaded libLiteRt.so")
                } catch (e: Throwable) {
                    Log.w(TAG, "Native library load warning: ${e.message}")
                }
                nativeLoaded = true
            }
        }

        @Volatile
        private var INSTANCE: LiteRTLMManager? = null

        fun orderBackendNames(
            defaultOrder: List<String> = listOf("NPU", "GPU", "CPU"),
            preferred: String?
        ): List<String> {
            val normalized = preferred?.trim()?.uppercase().orEmpty()
            if (normalized.isEmpty()) return defaultOrder
            val match = defaultOrder.firstOrNull { it.equals(normalized, ignoreCase = true) }
                ?: return defaultOrder
            return listOf(match) + defaultOrder.filterNot { it.equals(match, ignoreCase = true) }
        }

        fun isContextOverflowError(error: Throwable): Boolean {
            val msg = error.message.orEmpty()
            return msg.contains("KV-cache", ignoreCase = true) ||
                msg.contains("kv cache", ignoreCase = true) ||
                msg.contains("out of range", ignoreCase = true) ||
                msg.contains("context length", ignoreCase = true) ||
                msg.contains("max_num_tokens", ignoreCase = true)
        }

        fun isRoleAlternationError(error: Throwable): Boolean {
            val msg = error.message.orEmpty()
            return msg.contains("Conversation roles must alternate", ignoreCase = true)
        }

        fun getInstance(context: Context): LiteRTLMManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRTLMManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        isEmbedding: Boolean = false,
        preferredBackend: String? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        ensureNativeLoaded()
        if (isInitialized) cleanup()

        try {
            currentSystemPrompt = systemPrompt?.takeIf { it.isNotBlank() }
            val orderedBackends = buildBackendList(preferredBackend)

            initializeWithFallback(modelPath, orderedBackends)
            isInitialized = true
            resetConversation()
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Init failed", e)
            Result.failure(e)
        }
    }

    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        val orderedNames = orderBackendNames(listOf("NPU", "GPU", "CPU"), preferred)
        return orderedNames.map { name ->
            when (name) {
                "NPU" -> BackendFactory("NPU", nativeLibraryDir = nativeLibraryDir) {
                    Backend.NPU(nativeLibraryDir = nativeLibraryDir)
                }
                "GPU" -> BackendFactory("GPU") { Backend.GPU() }
                else -> BackendFactory("CPU") { Backend.CPU() }
            }
        }
    }

    private fun configureNativeRuntime(nativeLibraryDir: String) {
        try {
            Os.setenv("LD_LIBRARY_PATH", nativeLibraryDir, true)
            Os.setenv("ADSP_LIBRARY_PATH", nativeLibraryDir, true)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to configure native environment paths: ${e.message}")
        }
    }

    private fun initializeWithFallback(modelPath: String, backends: List<BackendFactory>) {
        val modelFile = File(modelPath)
        if (!modelFile.exists() || !modelFile.canRead()) {
            throw IllegalStateException("Model file not found or unreadable: $modelPath")
        }

        var lastError: Throwable? = null
        for (factory in backends) {
            var candidate: Engine? = null
            try {
                if (factory.name == "NPU") {
                    factory.nativeLibraryDir?.let { configureNativeRuntime(it) }
                }
                val backend = factory.create()
                val config = EngineConfig(
                    modelPath = modelPath,
                    backend = backend,
                    visionBackend = backend,
                    audioBackend = Backend.CPU(),
                    maxNumImages = 1,
                    cacheDir = context.cacheDir.path
                )

                candidate = Engine(config)
                candidate.initialize()
                engine = candidate
                currentBackendName = factory.name
                return
            } catch (e: Throwable) {
                Log.w(TAG, "${factory.name} failed: ${e.message}")
                runCatching { candidate?.close() }
                lastError = e
            }
        }
        throw lastError ?: IllegalStateException("All backends failed")
    }

    fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text).map { it.toString() }
    }

    fun sendMultimodalMessage(
        text: String,
        imagePath: String?,
        audioBytes: ByteArray?
    ): Flow<String> {
        ensureConversation()
        val parts = mutableListOf<Content>()
        if (imagePath != null) parts.add(Content.ImageFile(imagePath))
        if (audioBytes != null) parts.add(Content.AudioBytes(audioBytes))
        parts.add(Content.Text(text))
        return conversation!!.sendMessageAsync(Contents.of(*parts.toTypedArray()))
            .map { it.toString() }
    }

    @Synchronized
    fun resetConversation(): Boolean {
        val activeEngine = engine ?: return false
        if (!isInitialized) return false
        return try {
            conversation?.close()
            conversation = null
            val config = ConversationConfig(
                systemInstruction = currentSystemPrompt?.let { Contents.of(it) },
                samplerConfig = SamplerConfig(
                    temperature = 0.7,
                    topK = 40,
                    topP = 0.9
                )
            )
            conversation = activeEngine.createConversation(config)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to reset conversation", e)
            conversation = null
            false
        }
    }

    private fun ensureConversation() {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        if (conversation == null) {
            resetConversation()
        }
    }

    fun isReady(): Boolean = isInitialized && engine != null

    fun getActiveBackendName(): String = currentBackendName

    @Synchronized
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
        } catch (_: Exception) {
        }
        conversation = null
        engine = null
        isInitialized = false
    }
}

private data class BackendFactory(
    val name: String,
    val nativeLibraryDir: String? = null,
    val create: () -> Backend
)

