package com.example.gemma3_on_device

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Singleton manager for LiteRT-LM Engine.
 * GPU and CPU only version. Strictly NO NPU.
 */
class LiteRTLMManager private constructor(private val context: Context) {
    
    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    private var isInitialized = false
    private var currentBackendName: String = "CPU"
    
    companion object {
        private const val TAG = "LiteRTLMManager"
        
        init {
            try {
                // Load LiteRt first
                System.loadLibrary("LiteRt")
                Log.i(TAG, "Loaded libLiteRt.so")
            } catch (e: Throwable) {
                Log.w(TAG, "Native library load warning: ${e.message}")
            }
        }

        @Volatile
        private var INSTANCE: LiteRTLMManager? = null
        
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
        if (isInitialized) cleanup()
        
        try {
            val backends = listOf(
                BackendFactory("GPU") { Backend.GPU() },
                BackendFactory("CPU") { Backend.CPU() }
            )
            
            initializeWithFallback(modelPath, backends)
            isInitialized = true
            Result.success(true)
        } catch (e: Throwable) {
            Log.e(TAG, "Init failed", e)
            Result.failure(e)
        }
    }

    private fun initializeWithFallback(modelPath: String, backends: List<BackendFactory>) {
        var lastError: Throwable? = null
        for (factory in backends) {
            try {
                val backend = factory.create()
                // For Gemma 4 Multimodal, we need to specify backends for each modality.
                // We use the same backend (GPU or CPU) for all modalities if possible.
                val config = EngineConfig(
                    modelPath = modelPath,
                    backend = backend,
                    maxNumImages = 1,
                    cacheDir = context.cacheDir.path
                )
                
                val newEngine = Engine(config)
                newEngine.initialize()
                engine = newEngine
                currentBackendName = factory.name
                return
            } catch (e: Throwable) {
                Log.w(TAG, "${factory.name} failed: ${e.message}")
                lastError = e
            }
        }
        throw lastError ?: Exception("All backends failed")
    }

    fun sendMessage(text: String): Flow<String> {
        ensureConversation()
        return conversation!!.sendMessageAsync(text).map { it.toString() }
    }

    private fun ensureConversation() {
        if (conversation == null) {
            val config = ConversationConfig(
                samplerConfig = SamplerConfig(
                    temperature = 0.7,
                    topK = 40,
                    topP = 0.9
                )
            )
            conversation = engine?.createConversation(config)
        }
    }

    fun getActiveBackendName() = currentBackendName
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
        } catch (e: Exception) {}
        conversation = null; engine = null; isInitialized = false
    }
}

private data class BackendFactory(val name: String, val create: () -> Backend)
