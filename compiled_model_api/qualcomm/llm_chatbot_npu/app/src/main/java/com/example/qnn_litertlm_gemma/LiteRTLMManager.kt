package com.example.qnn_litertlm_gemma

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Data class for model performance metrics
 */
data class PerformanceMetrics(
    val initializationTimeMs: Long = 0,
    val timeToFirstTokenMs: Long = 0,
    val tokensPerSecond: Double = 0.0,
    val activeBackend: String = "Unknown",
    val memoryUsageMb: Long = 0
)

/**
 * Singleton manager for LiteRT-LM Engine.
 * Handles model initialization, conversation management,
 * and multimodal message sending (text, image, audio).
 *
 * Backend fallback chain: NPU → GPU → CPU
 */
class LiteRTLMManager private constructor(private val context: Context) {
    
    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"
    private var nativeRuntimeConfigured = false
    
    companion object {
        private const val TAG = "LiteRTLMManager"
        
        init {
            Log.i(TAG, "Using LiteRT-LM native loader")
        }

        @Volatile
        private var INSTANCE: LiteRTLMManager? = null
        
        fun getInstance(context: Context): LiteRTLMManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRTLMManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Initialize the LiteRT-LM Engine with the specified model.
     * Uses NPU → GPU → CPU fallback chain.
     */
    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        isEmbedding: Boolean = false,
        preferredBackend: String? = null,
        supportsImage: Boolean = true,
        supportsAudio: Boolean = true
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Initializing for: $modelPath (preferred: $preferredBackend)")
        if (isInitialized) {
            cleanup()
        }
        
        try {
            if (isEmbedding) {
                Log.w(TAG, "Embedding mode not supported in this version")
                currentBackendName = "CPU"
            } else {
                // Build ordered backend list based on preference
                val backends = buildBackendList(preferredBackend)
                initializeEngineWithFallback(modelPath, backends, supportsImage, supportsAudio)
            }
            isInitialized = true
            Log.i(TAG, "Initialization SUCCEEDED on backend: $currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Initialization FAILED: ${e.message}", e)
            Result.failure(Exception(e))
        }
    }

    /**
     * Build an ordered list of backends to try.
     * NPU → GPU → CPU by default; shifts based on preference.
     */
    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        val allBackends = listOf(
            BackendFactory("NPU", nativeLibraryDir = nativeLibraryDir) {
                val libDir = nativeLibraryDir
                Log.i(TAG, "Using native library dir for NPU: $libDir")
                Backend.NPU(nativeLibraryDir = libDir)
            },
            BackendFactory("GPU") { Backend.GPU() },
            BackendFactory("CPU") { Backend.CPU() }
        )
        
        if (preferred == null) return allBackends
        
        val preferredUpper = preferred.uppercase()
        val preferredIdx = allBackends.indexOfFirst { it.name == preferredUpper }
        
        return if (preferredIdx > 0) {
            // Move preferred to front, keep rest in order
            listOf(allBackends[preferredIdx]) + allBackends.filterIndexed { i, _ -> i != preferredIdx }
        } else {
            allBackends
        }
    }

    /**
     * Try each backend in order; stop at the first one that works.
     */
    private fun initializeEngineWithFallback(
        modelPath: String,
        backends: List<BackendFactory>,
        supportsImage: Boolean,
        supportsAudio: Boolean
    ) {
        var lastError: Throwable? = null
        
        for (factory in backends) {
            try {
                Log.i(TAG, "Trying backend: ${factory.name}")
                initializeEngine(modelPath, factory, supportsImage, supportsAudio)
                Log.i(TAG, "Backend ${factory.name} SUCCEEDED")
                return
            } catch (e: Throwable) {
                Log.w(TAG, "Backend ${factory.name} failed: ${e.message}")
                lastError = e
            }
        }
        
        throw lastError ?: IllegalStateException("All backends failed")
    }

    private fun initializeEngine(
        modelPath: String,
        factory: BackendFactory,
        supportsImage: Boolean,
        supportsAudio: Boolean
    ) {
        val file = File(modelPath)
        if (!file.exists()) {
            throw java.io.FileNotFoundException("Model file not found at $modelPath")
        }
        if (!file.canRead()) {
            throw java.io.IOException("Model file not readable (permissions?)")
        }
        Log.i(TAG, "Model file: ${file.length()} bytes, backend: ${factory.name}")

        val libDir = factory.nativeLibraryDir ?: context.applicationInfo.nativeLibraryDir
        if (factory.name == "NPU") {
            configureNativeRuntime(libDir)
        }

        val backend = factory.create()

        Log.i(TAG, "Initializing Engine with backend: ${factory.name}")

        val visionBackend = if (!supportsImage) {
            null
        } else if (factory.name == "NPU") {
            Backend.NPU(nativeLibraryDir = libDir)
        } else {
            Backend.CPU()
        }
        val visionBackendName = if (!supportsImage) "none" else if (factory.name == "NPU") "NPU" else "CPU"
        val audioBackend = if (supportsAudio) Backend.CPU() else null
        val audioBackendName = if (supportsAudio) "CPU" else "none"
        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            // Multimodal LiteRT-LM packages require a vision executor for
            // image prompts. Keep vision on NPU for SM8750 NPU model runs.
            visionBackend = visionBackend,
            // The packaged LiteRT-LM audio executor supports CPU/GPU, but not NPU.
            // Use CPU for Gemma audio; leave it unset for image-only models.
            audioBackend = audioBackend,
            maxNumImages = 1,
            // Cache dir is CRITICAL for JIT compilation
            cacheDir = context.cacheDir.path
        )
        Log.i(
            TAG,
            "Engine config: text=${factory.name}, vision=$visionBackendName, audio=$audioBackendName, maxNumImages=1"
        )
        
        val startTime = System.currentTimeMillis()
        val candidateEngine = Engine(engineConfig)
        
        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Log.i(TAG, "Engine initialization SUCCEEDED in ${duration}ms")
            
            // Verify conversation works
            val testConv = candidateEngine.createConversation(ConversationConfig())
            testConv.close()
        } catch (e: Throwable) {
            val duration = System.currentTimeMillis() - startTime
            Log.e(TAG, "Engine initialization FAILED after ${duration}ms: ${e.message}")
            if (factory.name == "NPU" && e.message?.contains("TF_LITE_AUX") == true) {
                Log.e(TAG, "NPU missing AOT payload and JIT compilation failed or was not triggered.")
            }
            candidateEngine.close()
            throw e
        }

        engine = candidateEngine
        currentBackendName = factory.name
    }

    @Synchronized
    private fun configureNativeRuntime(nativeLibraryDir: String) {
        if (nativeRuntimeConfigured) {
            return
        }

        try {
            android.system.Os.setenv("LD_LIBRARY_PATH", nativeLibraryDir, true)
            android.system.Os.setenv("ADSP_LIBRARY_PATH", nativeLibraryDir, true)
            Log.i(TAG, "Set native library paths to $nativeLibraryDir")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to set native library paths: ${e.message}")
        }

        nativeRuntimeConfigured = true
    }

    /**
     * Start a new conversation.
     */
    fun startConversation(systemPrompt: String? = null) {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        
        val conversationConfig = ConversationConfig(
            systemInstruction = if (systemPrompt != null) Contents.of(systemPrompt) else null
        )
        Log.i(TAG, "Starting conversation with model/runtime default sampler on $currentBackendName")
        conversation = engine?.createConversation(conversationConfig)
    }

    /**
     * Send a text-only message and stream the response.
     */
    fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text).map { msg ->
            msg.extractText()
        }
    }

    /**
     * Send a multimodal message with optional image and/or audio.
     * @param text The text prompt
     * @param imagePath Optional path to an image file
     * @param audioPath Optional path to an audio file
     */
    fun sendMultimodalMessage(
        text: String,
        imagePath: String? = null,
        audioPath: String? = null
    ): Flow<String> {
        ensureConversation()
        
        val contentParts = mutableListOf<Content>()
        
        // Add image if provided
        if (imagePath != null) {
            val imageFile = File(imagePath)
            Log.i(
                TAG,
                "Adding image content: path=$imagePath exists=${imageFile.exists()} size=${imageFile.length()}"
            )
            contentParts.add(Content.ImageFile(imagePath))
        }
        
        // Add audio if provided
        if (audioPath != null) {
            val audioFile = File(audioPath)
            Log.i(
                TAG,
                "Adding audio content: path=$audioPath exists=${audioFile.exists()} size=${audioFile.length()}"
            )
            contentParts.add(Content.AudioFile(audioPath))
        }
        
        // Always add text
        contentParts.add(Content.Text(text))
        
        val contents = Contents.of(*contentParts.toTypedArray())
        
        return conversation!!.sendMessageAsync(contents).map { msg ->
            msg.extractText()
        }
    }

    private fun Message.extractText(): String {
        val text = contents.contents.joinToString(separator = "") { content ->
            when (content) {
                is Content.Text -> content.text
                else -> ""
            }
        }
        if (text.isEmpty()) {
            Log.w(TAG, "Received non-text or empty message chunk: $this")
        }
        return text
    }

    private fun ensureConversation() {
        if (!isInitialized || engine == null) {
            throw IllegalStateException("Engine not initialized.")
        }
        if (conversation == null) {
            startConversation()
        }
    }

    fun getActiveBackendName(): String = currentBackendName

    fun getMemoryUsageMb(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }
    
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
            conversation = null
            engine = null
            isInitialized = false
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}

/**
 * Factory for creating backends lazily.
 * This avoids constructing NPU/GPU backends that might throw
 * before we're ready to handle the exception.
 */
private data class BackendFactory(
    val name: String,
    val nativeLibraryDir: String? = null,
    val create: () -> Backend
)
