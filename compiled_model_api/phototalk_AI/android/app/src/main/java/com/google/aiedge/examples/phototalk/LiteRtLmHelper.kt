/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.phototalk

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File

class LiteRtLmHelper private constructor(private val context: Context) {

    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var isInitialized = false
    private var activeBackendName: String = "CPU"

    companion object {
        private const val TAG = "PhotoTalk_LiteRtLM"

        @Volatile
        private var INSTANCE: LiteRtLmHelper? = null

        fun getInstance(context: Context): LiteRtLmHelper {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRtLmHelper(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    suspend fun initializeEngine(modelPath: String): Result<Boolean> = withContext(Dispatchers.IO) {
        if (isInitialized && engine != null) {
            return@withContext Result.success(true)
        }

        try {
            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                return@withContext Result.failure(
                    IllegalArgumentException("Model file not found at: $modelPath")
                )
            }

            // Attempt initialization with GPU backend first, falling back to CPU
            val backends = listOf(
                Pair("GPU") { Backend.GPU() },
                Pair("CPU") { Backend.CPU() }
            )

            var lastException: Throwable? = null
            for ((name, backendProvider) in backends) {
                try {
                    Log.i(TAG, "Initializing LiteRT-LM Engine with $name backend...")
                    val config = EngineConfig(
                        modelPath = modelPath,
                        backend = backendProvider(),
                        cacheDir = context.cacheDir.path
                    )
                    val newEngine = Engine(config)
                    newEngine.initialize()

                    engine = newEngine
                    activeBackendName = name
                    isInitialized = true
                    Log.i(TAG, "LiteRT-LM Engine successfully initialized on $name.")
                    return@withContext Result.success(true)
                } catch (e: Throwable) {
                    Log.w(TAG, "Failed to initialize LiteRT-LM Engine on $name", e)
                    lastException = e
                }
            }

            Result.failure(lastException ?: IllegalStateException("All backends failed to initialize LiteRT-LM"))
        } catch (e: Throwable) {
            Log.e(TAG, "LiteRT-LM Initialization Exception", e)
            Result.failure(e)
        }
    }

    suspend fun startImageConversation(
        detectedLabel: String,
        confidencePct: Float
    ): Flow<String> = withContext(Dispatchers.IO) {
        closeConversation()
        val currentEngine = engine ?: throw IllegalStateException("LiteRT-LM Engine is not initialized")

        val systemPrompt = "You are PhotoTalk AI, a friendly and insightful visual assistant. " +
                "The user has uploaded an image that was classified as '$detectedLabel' (confidence: ${"%.1f".format(confidencePct)}%). " +
                "Introduce yourself, acknowledge the detected object, share an interesting fun fact about it, " +
                "and ask the user what they would like to know about it."

        val config = ConversationConfig(
            systemInstruction = Contents.of(systemPrompt),
            samplerConfig = SamplerConfig(
                temperature = 0.7,
                topK = 40,
                topP = 0.9
            )
        )

        val newConversation = currentEngine.createConversation(config)
        conversation = newConversation

        // Initial prompt to trigger greeting and topic introduction
        newConversation.sendMessageAsync("Hello! Tell me about the photo I just uploaded.").map { it.toString() }
    }

    fun sendChatMessage(userText: String): Flow<String> {
        val currentConv = conversation ?: return flow {
            emit("Error: Conversation is not active. Please select an image first.")
        }
        return currentConv.sendMessageAsync(userText).map { it.toString() }
    }

    fun getActiveBackend(): String = activeBackendName

    private fun closeConversation() {
        try {
            conversation?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing conversation", e)
        }
        conversation = null
    }

    fun cleanup() {
        closeConversation()
        try {
            engine?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing engine", e)
        }
        engine = null
        isInitialized = false
    }
}
