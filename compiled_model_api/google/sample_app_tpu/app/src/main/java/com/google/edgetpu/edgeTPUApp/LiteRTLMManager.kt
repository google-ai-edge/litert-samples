package com.google.edgetpu.edgeTPUApp

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
 * Singleton manager for LiteRT-LM Engine, tailored for Google Edge TPU and fallback.
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
        supportsAudio: Boolean = false
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        Log.i(TAG, "Initializing model: $modelPath (preferred: $preferredBackend)")
        if (isInitialized) {
            cleanup()
        }
        

        try {
            val backends = buildBackendList(preferredBackend)
            initializeEngineWithFallback(modelPath, backends, systemPrompt, supportsImage, supportsAudio)
            isInitialized = true
            Log.i(TAG, "Initialization SUCCEEDED on backend: $currentBackendName")
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Initialization FAILED: ${e.message}", e)
            Result.failure(Exception(e))
        }
    }

    private fun buildBackendList(preferred: String?): List<BackendFactory> {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        val npuBackend = BackendFactory("NPU") {
            Log.i(TAG, "Using NPU backend with nativeLibraryDir: $nativeLibraryDir")
            Backend.NPU(nativeLibraryDir = nativeLibraryDir)
        }
        val gpuBackend = BackendFactory("GPU") { Backend.GPU() }
        val cpuBackend = BackendFactory("CPU") { Backend.CPU() }
        
        return when (preferred?.uppercase()) {
            "NPU" -> listOf(npuBackend, gpuBackend, cpuBackend)
            "GPU" -> listOf(gpuBackend, cpuBackend)
            "CPU" -> listOf(cpuBackend, gpuBackend)
            else -> listOf(gpuBackend, cpuBackend)
        }
    }

    private fun initializeEngineWithFallback(
        modelPath: String,
        backends: List<BackendFactory>,
        systemPrompt: String?,
        supportsImage: Boolean,
        supportsAudio: Boolean
    ) {
        var lastError: Throwable? = null
        
        for (factory in backends) {
            try {
                Log.i(TAG, "Trying backend: ${factory.name}")
                initializeEngine(modelPath, factory, systemPrompt, supportsImage, supportsAudio)
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
        supportsAudio: Boolean
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

        val visionBackend = if (!supportsImage) {
            null
        } else if (factory.name == "NPU") {
            Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        } else if (factory.name == "GPU") {
            Backend.GPU()
        } else {
            Backend.CPU()
        }

        val audioBackend = if (supportsAudio) Backend.CPU() else null

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            visionBackend = visionBackend,
            audioBackend = audioBackend,
            maxNumImages = 1,
            cacheDir = context.cacheDir.path
        )
        
        val startTime = System.currentTimeMillis()
        val candidateEngine = Engine(engineConfig)
        
        try {
            candidateEngine.initialize()
            val duration = System.currentTimeMillis() - startTime
            Log.i(TAG, "Engine initialization SUCCEEDED in ${duration}ms")
            
            // Create conversation config with system prompt if provided
            val conversationConfig = ConversationConfig(
                systemInstruction = if (systemPrompt != null) Contents.of(systemPrompt) else null
            )
            conversation = candidateEngine.createConversation(conversationConfig)
        } catch (e: Throwable) {
            val duration = System.currentTimeMillis() - startTime
            Log.e(TAG, "Engine initialization FAILED after ${duration}ms: ${e.message}")
            try {
                candidateEngine.close()
            } catch (closeEx: Throwable) {
                Log.w(TAG, "Error closing candidate engine: ${closeEx.message}")
            }
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
            // Workaround for 0.13.1 bug: The C++ engine looks for the dispatch library 
            // in the model's parent directory because it drops the nativeLibraryDir parameter.
            // Since our model is in the cacheDir, we copy the dispatch library there!
            val sourceFile = File(nativeLibraryDir, "libLiteRtDispatch_GoogleTensor.so")
            val destFile = File(context.cacheDir, "libLiteRtDispatch_GoogleTensor.so")
            if (sourceFile.exists()) {
                Log.i(TAG, "Copying NPU dispatch library to model directory (workaround for 0.13.1)")
                sourceFile.copyTo(destFile, overwrite = true)
            } else {
                Log.w(TAG, "NPU dispatch library NOT found in $nativeLibraryDir")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to copy NPU dispatch library: ${e.message}")
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
