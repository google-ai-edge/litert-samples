package com.example.fastvlm.engine

import android.content.Context
import android.net.Uri
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.LogSeverity
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * InferenceEngine following the v1 reference architecture.
 * Optimized for NPU execution on the Samsung S25 Ultra (SM8850).
 */
class InferenceEngine(private val context: Context) {

    init {
        try {
            // Load QNN and LiteRT dependencies explicitly to ensure the Dispatch library finds them
            System.loadLibrary("QnnSystem")
            System.loadLibrary("QnnHtp")
            System.loadLibrary("LiteRt")
            System.loadLibrary("LiteRtDispatch_Qualcomm")
            System.loadLibrary("litertlm_jni")
            Log.i(TAG, "Loaded native NPU chain successfully")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load litertlm_jni", e)
        }
    }

    companion object {
        private const val TAG = "InferenceEngine"
    }

    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var currentBackend: String = "NPU"

    /**
     * Initializes the engine with the specified backend.
     * Default is NPU as requested by the user.
     */
    suspend fun initialize(backend: String = "NPU") {
        withContext(Dispatchers.IO) {
            close()
            
            currentBackend = backend
            val modelPath = ModelDownloader.getModelPath(context, backend)
            val nativeLibDir = context.applicationInfo.nativeLibraryDir
            
            Log.d(TAG, "Initializing v1 engine on $backend...")
            Log.d(TAG, "Model path: $modelPath")
            
            // Set logging severity
            Engine.setNativeMinLogSeverity(LogSeverity.VERBOSE)

            // Configure backends using standard v1 API
            val backendEnum = when (backend) {
                "NPU" -> Backend.NPU(nativeLibraryDir = nativeLibDir)
                "GPU" -> Backend.GPU()
                else -> Backend.CPU()
            }

            // Enforce NPU for vision if main backend is NPU
            val visionBackendEnum = if (backend == "NPU") {
                Backend.NPU(nativeLibraryDir = nativeLibDir)
            } else {
                backendEnum
            }

            val config = EngineConfig(
                modelPath = modelPath,
                backend = backendEnum,
                visionBackend = visionBackendEnum,
            )

            try {
                // The Engine(config) constructor handles internal library loading in v1
                engine = Engine(config)
                engine!!.initialize()
                conversation = engine!!.createConversation()
                Log.i(TAG, "Engine initialized successfully on $backend")
            } catch (e: Exception) {
                Log.e(TAG, "Engine initialization failed on $backend", e)
                throw e
            }
        }
    }

    fun sendTextMessage(text: String): Flow<String> = flow {
        val conv = conversation ?: throw IllegalStateException("Engine not initialized")
        val message = Message.user(text)

        conv.sendMessageAsync(message).collect { responseMessage ->
            val textChunk = responseMessage.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }
            emit(textChunk)
        }
    }.flowOn(Dispatchers.IO)

    fun sendMultimodalMessage(text: String, imageUri: Uri): Flow<String> = flow {
        val conv = conversation ?: throw IllegalStateException("Engine not initialized")
        val imagePath = copyImageToTemp(imageUri)

        val message = Message.user(
            Contents.of(
                Content.ImageFile(imagePath),
                Content.Text(text)
            )
        )

        conv.sendMessageAsync(message).collect { responseMessage ->
            val textChunk = responseMessage.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }
            emit(textChunk)
        }
    }.flowOn(Dispatchers.IO)

    suspend fun resetConversation() {
        withContext(Dispatchers.IO) {
            conversation?.close()
            conversation = engine?.createConversation()
            Log.d(TAG, "Conversation reset")
        }
    }

    private fun copyImageToTemp(uri: Uri): String {
        val tempFile = File(context.cacheDir, "input_image_${System.currentTimeMillis()}.jpg")
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalArgumentException("Could not open image URI")
        return tempFile.absolutePath
    }

    fun isInitialized(): Boolean = engine != null
    fun getCurrentBackend(): String = currentBackend

    fun close() {
        conversation?.close()
        engine?.close()
        conversation = null
        engine = null
    }
}
