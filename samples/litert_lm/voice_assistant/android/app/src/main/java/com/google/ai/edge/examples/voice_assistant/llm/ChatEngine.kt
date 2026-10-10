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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// litertlm/src/main/kotlin/io/github/johnrocky/hfmodels/litertlm/LiteRtLmChatModel.kt
// litertlm/src/main/kotlin/io/github/johnrocky/hfmodels/litertlm/LiteRtLmHandler.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.llm

import android.util.Log
import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.handOver
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Channel
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.ai.edge.litertlm.ToolProvider
import java.io.File
import java.util.concurrent.CancellationException as JavaCancellationException
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.Channel as KChannel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Gemma 4 E2B on LiteRT-LM: one [Engine] on the GPU and, per request, one [Conversation] with the
 * system instruction and the tools (`automaticToolCalling = false`: the runtime parses the calls
 * into `Message.toolCalls` and the app runs them). Conversation creation and release run on one
 * dedicated thread; generation callbacks arrive on the runtime's own thread and are bridged into a
 * bounded channel (no chunk is dropped: an overflow cancels the native side and ends the flow with
 * `SLOW_CONSUMER`). One generation at a time per engine.
 */
class ChatEngine private constructor(private val engine: Engine) : AutoCloseable {
  private val executor =
    Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }
  private val nativeDispatcher = executor.asCoroutineDispatcher()
  private val scope = CoroutineScope(SupervisorJob() + nativeDispatcher)
  private val conversations = CopyOnWriteArrayList<ChatConversation>()
  private val generating = AtomicReference<ChatConversation?>(null)
  private val closing = AtomicBoolean(false)
  private val closed = CompletableDeferred<Unit>()
  private val closeLock = Mutex()

  @Volatile private var unusable: String? = null

  /**
   * A new conversation: [systemInstruction], [tools], greedy decoding and the bundle's thought
   * channel. Each request of the voice loop opens its own and closes it at the end.
   */
  suspend fun createConversation(
    systemInstruction: String,
    tools: List<ToolProvider>,
  ): ChatConversation {
    checkUsable()
    val config =
      ConversationConfig(
        systemInstruction = Contents.of(systemInstruction),
        tools = tools,
        samplerConfig = GREEDY,
        automaticToolCalling = false,
        channels = CHANNELS,
      )
    // Made on the native thread and handed over: a caller cancelled during the native call gets
    // nothing, and the conversation is closed instead of left open outside [conversations].
    return handOver(nativeDispatcher, { newConversation(config) }) { it.closeAndJoin() }
  }

  /** On the native thread. */
  private fun newConversation(config: ConversationConfig): ChatConversation {
    checkUsable()
    val conversation =
      try {
        engine.createConversation(config)
      } catch (t: Throwable) {
        throw VoiceAssistantException(
          "INITIALIZATION_FAILED",
          "createConversation failed: ${t.javaClass.simpleName}: ${t.message}",
          t,
        )
      }
    return ChatConversation(conversation).also { conversations += it }
  }

  override fun close() {
    if (closing.compareAndSet(false, true)) {
      scope.launch {
        doClose()
        executor.shutdown()
      }
    }
  }

  /** Closes every conversation (each waits for its generation to stop), then the engine. */
  suspend fun closeAndJoin() {
    closing.set(true)
    withContext(NonCancellable) {
      if (!closed.isCompleted && !executor.isShutdown) {
        withContext(nativeDispatcher) { doClose() }
      }
      closed.await()
      executor.shutdown()
    }
  }

  private suspend fun doClose(): Unit =
    closeLock.withLock {
      if (closed.isCompleted) {
        return@withLock
      }
      // Children first: each waits for its native work to stop (10 s cap), then the Engine.
      for (c in conversations.toList()) {
        runCatching { c.closeAndJoin() }
          .onFailure { Log.w(TAG, "conversation close during engine close: ${it.message}") }
      }
      if (unusable == null) {
        runCatching { engine.close() }.onFailure { Log.w(TAG, "engine.close: ${it.message}", it) }
      } else {
        Log.e(TAG, "engine left unusable ($unusable): not released while a native call may hold it")
      }
      closed.complete(Unit)
    }

  private fun checkUsable() {
    unusable?.let { throw VoiceAssistantException("NATIVE_STOP_TIMEOUT", "engine unusable: $it") }
    if (closing.get()) {
      throw VoiceAssistantException("MODEL_CLOSED", "the chat engine is closing or closed")
    }
  }

  private fun markUnusable(reason: String) {
    if (unusable == null) {
      unusable = reason
      Log.e(TAG, "NATIVE_STOP_TIMEOUT: $reason; the engine is not released")
    }
  }

  /**
   * One LiteRT-LM conversation. [send] streams one model turn; a cancelled or failed turn leaves
   * the conversation unusable (open a new one).
   */
  inner class ChatConversation internal constructor(private val conversation: Conversation) {
    private val closedDeferred = CompletableDeferred<Unit>()

    /** Completed by the runtime's onDone / onError of the generation in flight. */
    @Volatile private var nativeDone: CompletableDeferred<Unit>? = null

    @Volatile private var cancelRequested = false

    @Volatile private var invalid = false

    @Volatile private var closingConversation = false

    /**
     * Sends [message] (`Message.user(text)`, or `Message.tool(Contents.of(responses))` with one
     * `Content.ToolResponse` per call of the previous turn) and streams the model's turn: text in
     * `contents`, reasoning in `channels`, calls in `toolCalls`. [thinking] asks the model to
     * reason in its thought channel. At most [MAX_OUTPUT_TOKENS] tokens. Collect it once.
     */
    fun send(message: Message, thinking: Boolean): Flow<Message> {
      val collected = AtomicBoolean(false)
      return flow {
        if (!collected.compareAndSet(false, true)) {
          throw VoiceAssistantException("INFERENCE_FAILED", "this Flow was already collected")
        }
        if (invalid) {
          throw VoiceAssistantException(
            "SESSION_INVALIDATED",
            "this conversation was cancelled or failed; open a new one",
          )
        }
        if (closingConversation) {
          throw VoiceAssistantException("MODEL_CLOSED", "this conversation is closed")
        }
        checkUsable()
        if (!generating.compareAndSet(null, this@ChatConversation)) {
          throw VoiceAssistantException(
            "MODEL_BUSY",
            "another generation is running; one at a time per engine",
          )
        }
        cancelRequested = false
        val done = CompletableDeferred<Unit>().also { nativeDone = it }
        val channel = KChannel<Message>(capacity = STREAM_BUFFER_CHUNKS)
        val bytes = AtomicLong(0)
        val overflow = AtomicBoolean(false)
        val callback =
          object : MessageCallback {
            override fun onMessage(message: Message) {
              if (cancelRequested || overflow.get()) {
                return
              }
              val text =
                message.contents.contents.sumOf { (it as? Content.Text)?.text?.length ?: 0 }
              val size = text.toLong() + message.channels.values.sumOf { it.length.toLong() }
              val sent = channel.trySend(message)
              if (!sent.isSuccess || bytes.addAndGet(size) > STREAM_BUFFER_BYTES) {
                if (overflow.compareAndSet(false, true)) {
                  // The collector is too far behind: stop the native side; the flow ends with
                  // SLOW_CONSUMER.
                  runCatching { conversation.cancelProcess() }
                  channel.close(
                    VoiceAssistantException(
                      "SLOW_CONSUMER",
                      "the collector fell more than $STREAM_BUFFER_CHUNKS chunks or " +
                        "$STREAM_BUFFER_BYTES bytes behind; generation cancelled",
                    )
                  )
                }
              }
            }

            override fun onDone() {
              done.complete(Unit)
              channel.close()
            }

            override fun onError(throwable: Throwable) {
              done.complete(Unit)
              if (throwable is JavaCancellationException || throwable is CancellationException) {
                channel.close()
              } else {
                channel.close(throwable)
              }
            }
          }
        var completedNormally = false
        try {
          try {
            conversation.sendMessageAsync(
              message,
              callback,
              maxOutputToken = MAX_OUTPUT_TOKENS,
              thinkingConfig = ThinkingConfig(enableThinking = thinking, thinkingTokenBudget = -1),
            )
          } catch (t: Throwable) {
            done.complete(Unit)
            throw VoiceAssistantException(
              "INFERENCE_FAILED",
              "sendMessageAsync failed: ${t.javaClass.simpleName}: ${t.message}",
              t,
            )
          }
          for (m in channel) {
            emit(m)
          }
          completedNormally = true
        } catch (t: Throwable) {
          if (t is CancellationException) {
            cancelNative("collector cancelled")
            throw t
          }
          Log.w(TAG, "generation ended: ${t.message}")
          if (t is VoiceAssistantException) {
            throw t
          }
          throw VoiceAssistantException(
            "INFERENCE_FAILED",
            "generation failed: ${t.javaClass.simpleName}: ${t.message}",
            t,
          )
        } finally {
          // Never release the engine's generation slot before the runtime confirmed it stopped.
          withContext(NonCancellable) {
            withTimeoutOrNull(NATIVE_STOP_MS) { done.await() }
              ?: markUnusable("generation did not stop within $NATIVE_STOP_MS ms")
          }
          generating.compareAndSet(this@ChatConversation, null)
          if (unusable != null || !completedNormally || cancelRequested) {
            invalid = true
          }
        }
      }
    }

    private fun cancelNative(why: String) {
      if (cancelRequested) {
        return
      }
      cancelRequested = true
      runCatching { conversation.cancelProcess() }
        .onFailure { Log.w(TAG, "cancelProcess ($why): ${it.message}") }
    }

    /** Stops the generation in flight (no-op when idle). */
    fun cancel() {
      if (generating.get() === this) {
        cancelNative("cancel()")
      }
    }

    /** Waits until the native conversation is released (its generation stopped first). */
    suspend fun closeAndJoin() {
      if (closedDeferred.isCompleted) {
        return
      }
      withContext(NonCancellable) {
        if (generating.get() === this@ChatConversation) {
          cancelNative("close()")
        }
        closingConversation = true
        val pending = nativeDone
        if (pending != null && !pending.isCompleted) {
          withTimeoutOrNull(NATIVE_STOP_MS) { pending.await() }
            ?: markUnusable("conversation close: generation did not stop in $NATIVE_STOP_MS ms")
        }
        if (unusable == null) {
          withContext(nativeDispatcher) {
            runCatching { conversation.close() }
              .onFailure {
                if (it !is IllegalStateException) {
                  Log.w(TAG, "conversation close: ${it.message}")
                }
              }
          }
        }
        conversations.remove(this@ChatConversation)
        closedDeferred.complete(Unit)
      }
    }
  }

  companion object {
    private const val TAG = "VoiceAssistant"
    private const val THREAD = "voice-assistant-litertlm"
    private const val STREAM_BUFFER_CHUNKS = 1024
    private const val STREAM_BUFFER_BYTES = 8L * 1024 * 1024
    private const val NATIVE_STOP_MS = 10_000L

    /** The output cap of a turn for a model that does not reason by default (Gemma 4). */
    const val MAX_OUTPUT_TOKENS = 256

    /** Greedy decoding: the same prompt on the same model state gives the same answer. */
    private val GREEDY = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0)

    /**
     * Gemma 4's thought channel, as the bundle declares it in its own header: reasoning streams
     * into `Message.channels["thought"]`, never into the text. Gemma 4 does not reason unless
     * thinking is enabled.
     */
    private val CHANNELS = listOf(Channel("thought", "<|channel>thought\n", "<channel|>"))

    /**
     * Initializes LiteRT-LM on the GPU for [modelFile] (a `.litertlm` bundle), with the runtime's
     * caches in [cacheDir] (created when missing). A failure is INITIALIZATION_FAILED; there is no
     * fallback to the CPU.
     */
    suspend fun open(modelFile: File, cacheDir: File): ChatEngine {
      // The native initialize is not interrupted: a caller cancelled meanwhile gets nothing, and
      // the engine is closed instead of dropped.
      return handOver(Dispatchers.IO, { initialize(modelFile, cacheDir) }) { it.closeAndJoin() }
    }

    private fun initialize(modelFile: File, cacheDir: File): ChatEngine {
      check(cacheDir.isDirectory || cacheDir.mkdirs()) { "Cannot create $cacheDir" }
      val config =
        EngineConfig(
          modelPath = modelFile.absolutePath,
          backend = Backend.GPU(),
          cacheDir = cacheDir.absolutePath,
        )
      val engine = Engine(config)
      val t0 = System.nanoTime()
      try {
        engine.initialize()
      } catch (t: Throwable) {
        runCatching { engine.close() }
        throw VoiceAssistantException(
          "INITIALIZATION_FAILED",
          "Engine.initialize() failed on the GPU: ${t.javaClass.simpleName}: ${t.message}",
          t,
        )
      }
      Log.i(TAG, "litertlm initialize_ms=${(System.nanoTime() - t0) / 1_000_000} on gpu")
      return ChatEngine(engine)
    }
  }
}
