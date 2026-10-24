package com.example.gemma3_npu

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException

/**
 * Data class for model performance metrics.
 */
data class PerformanceMetrics(
    val initializationTimeMs: Long = 0,
    val timeToFirstTokenMs: Long = 0,
    val tokensPerSecond: Double = 0.0,
    val activeBackend: String = "Unknown",
    val memoryUsageMb: Long = 0
)

/**
 * Singleton manager for LiteRT-LM Engine on Gemma 3 1B NPU.
 * Handles model initialization, system prompt propagation, and KV-cache session resets.
 *
 * Backend fallback chain: NPU -> GPU -> CPU
 */
class LiteRTLMManager private constructor(private val context: Context) {

    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"
    private var currentSystemPrompt: String? = null
    private var nativeRuntimeConfigured = false

    companion object {
        private const val TAG = "LiteRTLMManager"

        @Volatile
        private var INSTANCE: LiteRTLMManager? = null

        fun getInstance(context: Context): LiteRTLMManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRTLMManager(context.applicationContext).also { INSTANCE = it }
            }
        }

        internal fun orderBackendNames(
            defaultOrder: List<String>,
            preferred: String?
        ): List<String> {
            if (preferred.isNullOrBlank()) return defaultOrder
            val target = preferred.trim().uppercase()
            val idx = defaultOrder.indexOfFirst { it.equals(target, ignoreCase = true) }
            return if (idx > 0) {
                listOf(defaultOrder[idx]) + defaultOrder.filterIndexed { i, _ -> i != idx }
            } else {
                defaultOrder
            }
        }

        internal fun isContextOverflowError(error: Throwable?): Boolean {
            val msg = error?.message?.lowercase() ?: return false
            return msg.contains("kv-cache update out of range") ||
                msg.contains("kv cache") ||
                msg.contains("max_num_tokens") ||
                msg.contains("context length")
        }

        internal fun isRoleAlternationError(error: Throwable?): Boolean {
            val msg = error?.message?.lowercase() ?: return false
            return msg.contains("conversation roles must alternate")
        }
    }

    /**
     * Initialize the LiteRT-LM Engine with the specified model.
     * Uses NPU -> GPU -> CPU fallback chain.
     */
    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        isEmbedding: Boolean = false,
        preferredBackend: String? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Initializing model: $modelPath (preferred backend: $preferredBackend)")
        if (isInitialized) {
            cleanup()
        }

        try {
            currentSystemPrompt = systemPrompt?.takeIf { it.isNotBlank() }
            if (isEmbedding) {
                Log.w(TAG, "Embedding mode is not supported in this sample")
                currentBackendName = "CPU"
            } else {
                val backends = buildBackendList(preferredBackend)
                initializeEngineWithFallback(modelPath, backends)
            }
            isInitialized = true
            startConversation(currentSystemPrompt)
            Log.i(TAG, "Initialization succeeded on backend: $currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Initialization failed: ${e.message}", e)
            Result.failure(Exception(e))
        }
    }

    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        val byName = mapOf(
            "NPU" to BackendFactory("NPU", nativeLibraryDir = nativeLibraryDir) {
                Log.i(TAG, "Using native library dir for NPU: $nativeLibraryDir")
                Backend.NPU(nativeLibraryDir = nativeLibraryDir)
            },
            "GPU" to BackendFactory("GPU") { Backend.GPU() },
            "CPU" to BackendFactory("CPU") { Backend.CPU() }
        )
        return orderBackendNames(listOf("NPU", "GPU", "CPU"), preferred)
            .mapNotNull { byName[it] }
    }

    private fun initializeEngineWithFallback(modelPath: String, backends: List<BackendFactory>) {
        var lastError: Throwable? = null

        for (factory in backends) {
            try {
                Log.i(TAG, "Trying backend: ${factory.name}")
                initializeEngine(modelPath, factory)
                Log.i(TAG, "Backend ${factory.name} initialized")
                return
            } catch (e: Throwable) {
                Log.w(TAG, "Backend ${factory.name} failed: ${e.message}")
                lastError = e
            }
        }

        throw lastError ?: IllegalStateException("All backends failed")
    }

    private fun initializeEngine(modelPath: String, factory: BackendFactory) {
        val file = File(modelPath)
        if (!file.exists()) {
            throw FileNotFoundException("Model file not found at $modelPath")
        }
        if (!file.canRead()) {
            throw IOException("Model file is not readable at $modelPath")
        }
        Log.i(TAG, "Model file: ${file.length()} bytes, backend: ${factory.name}")

        val libDir = factory.nativeLibraryDir ?: context.applicationInfo.nativeLibraryDir
        if (factory.name == "NPU") {
            configureNativeRuntime(libDir)
        }

        val backend = factory.create()
        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            cacheDir = context.cacheDir.path
        )

        val startTime = System.currentTimeMillis()
        val candidateEngine = Engine(engineConfig)

        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Log.i(TAG, "Engine initialized in ${duration}ms")

            val testConv = candidateEngine.createConversation(ConversationConfig())
            testConv.close()
        } catch (e: Throwable) {
            candidateEngine.close()
            throw e
        }

        engine = candidateEngine
        currentBackendName = factory.name
    }

    @Synchronized
    private fun configureNativeRuntime(nativeLibraryDir: String) {
        if (nativeRuntimeConfigured) return
        try {
            android.system.Os.setenv("LD_LIBRARY_PATH", nativeLibraryDir, true)
            android.system.Os.setenv("ADSP_LIBRARY_PATH", nativeLibraryDir, true)
            Log.i(TAG, "Configured native QNN paths to $nativeLibraryDir")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to configure native runtime environment: ${e.message}")
        }
        nativeRuntimeConfigured = true
    }

    /**
     * Starts or replaces the active conversation while preserving the configured system prompt.
     */
    @Synchronized
    fun startConversation(systemPrompt: String? = currentSystemPrompt) {
        val activeEngine = engine
        if (!isInitialized || activeEngine == null) {
            throw IllegalStateException("Engine not initialized.")
        }

        try {
            conversation?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Closing previous conversation threw: ${e.message}")
        }
        conversation = null

        val effectivePrompt = systemPrompt?.takeIf { it.isNotBlank() }
        val conversationConfig = ConversationConfig(
            systemInstruction = effectivePrompt?.let { Contents.of(it) }
        )
        Log.i(TAG, "Starting fresh conversation on $currentBackendName")
        conversation = activeEngine.createConversation(conversationConfig)
    }

    /**
     * Resets the conversation session and static NPU KV-cache without reloading the engine.
     */
    @Synchronized
    fun resetConversation(): Boolean {
        if (!isInitialized || engine == null) return false
        return try {
            startConversation(currentSystemPrompt)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to reset conversation: ${e.message}", e)
            conversation = null
            false
        }
    }

    fun isReady(): Boolean = isInitialized && engine != null

    /**
     * Send a text prompt and stream the response chunks.
     */
    fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text).map { msg ->
            msg.extractText()
        }
    }

    private fun Message.extractText(): String {
        return contents.contents.joinToString(separator = "") { content ->
            when (content) {
                is Content.Text -> content.text
                else -> ""
            }
        }
    }

    private fun ensureConversation() {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        if (conversation == null) {
            startConversation(currentSystemPrompt)
        }
    }

    fun getActiveBackendName(): String = currentBackendName

    fun getMemoryUsageMb(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }

    @Synchronized
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        } finally {
            conversation = null
            engine = null
            isInitialized = false
        }
    }
}

private data class BackendFactory(
    val name: String,
    val nativeLibraryDir: String? = null,
    val create: () -> Backend
)
