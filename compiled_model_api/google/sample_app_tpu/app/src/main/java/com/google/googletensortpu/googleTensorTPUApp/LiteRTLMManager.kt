package com.google.googletensortpu.googleTensorTPUApp

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
 * Singleton manager for LiteRT-LM Engine, tailored for Google Tensor TPU and fallback.
 * Backend fallback chain: NPU (TPU) → GPU → CPU
 */
class LiteRTLMManager private constructor(private val context: Context) {
    
    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"
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
    }
    
    /**
     * Initialize the LiteRT-LM Engine with the specified model.
     * Uses NPU → GPU → CPU fallback chain.
     */
    suspend fun initialize(
        modelPath: String,
        systemPrompt: String? = null,
        preferredBackend: String? = null,
        supportsImage: Boolean = true,
        supportsAudio: Boolean = false,
        maxContext: Int = 2048,
        initialMessages: List<Message>? = null
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Initializing model: $modelPath (preferred: $preferredBackend)")
        if (isInitialized) {
            cleanup()
        }
        

        try {
            val backends = buildBackendList(preferredBackend)
            initializeEngineWithFallback(modelPath, backends, systemPrompt, supportsImage, supportsAudio, maxContext, initialMessages)
            isInitialized = true
            Log.i(TAG, "Initialization SUCCEEDED on backend: $currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Initialization FAILED: ${e.message}", e)
            Result.failure(Exception(e))
        }
    }

    private fun isPackageInstalled(packageName: String): Boolean {
        return try {
            context.packageManager.getPackageInfo(packageName, 0)
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        
        val isNpuAvailable = isPackageInstalled("com.google.android.aicore")
        Log.i(TAG, "AICore package (com.google.android.aicore) installed: $isNpuAvailable")
        
        val npuBackend = BackendFactory("NPU") {
            Log.i(TAG, "Using NPU backend with nativeLibraryDir: $nativeLibraryDir")
            Backend.NPU(nativeLibraryDir = nativeLibraryDir)
        }
        val gpuBackend = BackendFactory("GPU") { Backend.GPU() }
        val cpuBackend = BackendFactory("CPU") { Backend.CPU() }
        
        val list = when (preferred?.uppercase()) {
            "NPU" -> listOf(npuBackend, gpuBackend, cpuBackend)
            "GPU" -> listOf(gpuBackend, cpuBackend)
            "CPU" -> listOf(cpuBackend, gpuBackend)
            else -> listOf(gpuBackend, cpuBackend)
        }
        
        return if (!isNpuAvailable) {
            Log.w(TAG, "NPU backend is not available on this device (AICore is missing). Filtering out NPU from fallback chain.")
            list.filter { it.name != "NPU" }
        } else {
            list
        }
    }

    private fun initializeEngineWithFallback(
        modelPath: String,
        backends: List<BackendFactory>,
        systemPrompt: String?,
        supportsImage: Boolean,
        supportsAudio: Boolean,
        maxContext: Int,
        initialMessages: List<Message>?
    ) {
        var lastError: Throwable? = null
        
        for (factory in backends) {
            try {
                Log.i(TAG, "Trying backend: ${factory.name}")
                initializeEngine(modelPath, factory, systemPrompt, supportsImage, supportsAudio, maxContext, initialMessages)
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
        systemPrompt: String?,
        supportsImage: Boolean,
        supportsAudio: Boolean,
        maxContext: Int,
        initialMessages: List<Message>?
    ) {
        val file = File(modelPath)
        if (!file.exists()) {
            throw java.io.FileNotFoundException("Model file not found at $modelPath")
        }
        if (!file.canRead()) {
            throw java.io.IOException("Model file not readable")
        }

        if (factory.name == "NPU") {
            configureNativeRuntime(context.applicationInfo.nativeLibraryDir)
        }

        val backend = factory.create()

        val visionBackend = if (factory.name == "NPU") {
            Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        } else if (factory.name == "GPU") {
            Backend.GPU()
        } else {
            Backend.CPU()
        }

        val audioBackend = Backend.CPU()

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            visionBackend = visionBackend,
            audioBackend = audioBackend,
            maxNumTokens = maxContext,
            maxNumImages = 1,
            cacheDir = context.cacheDir.path
        )
        
        val startTime = System.currentTimeMillis()
        var candidateEngine = Engine(engineConfig)
        
        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Log.i(TAG, "Engine initialization SUCCEEDED in ${duration}ms")
        } catch (e: Throwable) {
            val duration = System.currentTimeMillis() - startTime
            Log.e(TAG, "Engine initialization FAILED after ${duration}ms: ${e.message}")
            try {
                candidateEngine.close()
            } catch (closeEx: Throwable) {
                Log.w(TAG, "Error closing candidate engine: ${closeEx.message}")
            }
            
            Log.i(TAG, "Retrying initialization as text-only model without vision/audio backend...")
            val textOnlyConfig = EngineConfig(
                modelPath = modelPath,
                backend = factory.create(),
                visionBackend = null,
                audioBackend = null,
                maxNumTokens = maxContext,
                maxNumImages = null,
                cacheDir = context.cacheDir.path
            )
            candidateEngine = Engine(textOnlyConfig)
            try {
                candidateEngine.initialize()
                Log.i(TAG, "Text-only Engine initialization SUCCEEDED")
            } catch (retryEx: Throwable) {
                try {
                    candidateEngine.close()
                } catch (closeEx: Throwable) {
                    Log.w(TAG, "Error closing fallback candidate engine: ${closeEx.message}")
                }
                throw retryEx
            }
        }

        try {
            // Create conversation config with system prompt if provided
            val conversationConfig = ConversationConfig(
                systemInstruction = if (systemPrompt != null) Contents.of(systemPrompt) else null,
                initialMessages = initialMessages ?: emptyList()
            )
            conversation = candidateEngine.createConversation(conversationConfig)
        } catch (convEx: Throwable) {
            try {
                candidateEngine.close()
            } catch (closeEx: Throwable) {
                Log.w(TAG, "Error closing candidate engine after conversation failure: ${closeEx.message}")
            }
            throw convEx
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
            System.loadLibrary("LiteRtDispatch_GoogleTensor")
            Log.i(TAG, "Successfully loaded libLiteRtDispatch_GoogleTensor.so via System.loadLibrary")
        } catch (e: Throwable) {
            Log.w(TAG, "Failed to load LiteRtDispatch_GoogleTensor: ${e.message}")
        }

        try {
            val ldPath = android.system.Os.getenv("LD_LIBRARY_PATH") ?: ""
            val newLdPath = if (ldPath.isEmpty()) nativeLibraryDir else "$nativeLibraryDir:$ldPath"
            android.system.Os.setenv("LD_LIBRARY_PATH", newLdPath, true)
            Log.i(TAG, "Set LD_LIBRARY_PATH to $newLdPath")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to set LD_LIBRARY_PATH: ${e.message}")
        }

        nativeRuntimeConfigured = true
    }
    /**
     * Send a text-only message and stream the response.
     */
    fun sendMessage(text: String): Flow<String> {
        val conv = conversation ?: throw IllegalStateException("Engine/Conversation not initialized.")
        return conv.sendMessageAsync(text).map { msg ->
            msg.extractText()
        }
    }

    /**
     * Send a multimodal message with optional image and/or audio files.
     */
    fun sendMultimodalMessage(
        text: String,
        imagePath: String? = null,
        audioPath: String? = null
    ): Flow<String> {
        val conv = conversation ?: throw IllegalStateException("Engine/Conversation not initialized.")
        
        val contentParts = mutableListOf<Content>()
        
        if (imagePath != null) {
            contentParts.add(Content.ImageFile(imagePath))
        }
        
        if (audioPath != null) {
            contentParts.add(Content.AudioFile(audioPath))
        }
        
        contentParts.add(Content.Text(text))
        
        val contents = Contents.of(*contentParts.toTypedArray())
        
        return conv.sendMessageAsync(contents).map { msg ->
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

    fun getActiveBackendName(): String = currentBackendName

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
    val create: () -> Backend
)
