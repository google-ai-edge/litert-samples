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

package com.google.ai.edge.examples.click_or_code

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.util.concurrent.CancellationException as JavaCancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/** Where the model is between steps. */
sealed class EngineState {
  data object Idle : EngineState()

  /** The Engine is being made. */
  data object Preparing : EngineState()

  data class Ready(val engineMs: Long) : EngineState()

  data object Deciding : EngineState()

  /** The last attempt failed; the next step tries again. */
  data class Failed(val message: String) : EngineState()
}

/** The runtime's own counters of one step. */
data class Stages(
  /** From sending the message to the first token (the picture's prefill and the first decode). */
  val timeToFirstTokenS: Double,
  val prefillTokens: Int,
  val prefillTokensPerS: Double,
  val decodeTokens: Int,
  val decodeTokensPerS: Double,
  val marks: Map<String, Double>,
)

/** What a step reports while it runs, in order. */
sealed class StepEvent {
  data object Waiting : StepEvent()

  data class Rendered(val prompt: String) : StepEvent()

  data class Started(val waitMs: Long, val engineMs: Long, val conversationMs: Long) : StepEvent()

  data class Text(val text: String) : StepEvent()

  data class Finished(val stages: Stages?) : StepEvent()
}

/**
 * Mage-VL-Cowork on LiteRT-LM: the text decoder on the GPU, the vision encoder on the CPU (on the
 * Galaxy S26 the file's vision encoder does not run on the GPU; on a Pixel 11 Pro the CPU is also
 * the measured form). One Engine for the app, one new Conversation per inference: the decoder has no
 * state that crosses conversations, so an inference costs the same on a used Engine. Everything
 * native runs on one thread.
 */
class StepEngine(
  private val modelFile: File,
  private val cacheDir: File,
  private val systemPrompt: String,
  private val memory: MemoryCheck = MemoryCheck(),
) {
  private val nativeThread =
    Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }
  private val native = nativeThread.asCoroutineDispatcher()
  private val scope = CoroutineScope(SupervisorJob() + native)
  private val stepping = AtomicBoolean(false)
  private val lock = Any()
  private var loading: Deferred<Engine>? = null
  private var engine: Engine? = null
  private var engineMs = 0L
  private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
  @Volatile private var inFlight: Conversation? = null
  @Volatile private var closed = false
  @Volatile private var unusable: String? = null
  @Volatile private var stopAsked = false

  val state: StateFlow<EngineState> = _state.asStateFlow()

  /** Starts making the Engine unless it is made or being made. */
  fun prepare() {
    pending()
  }

  private fun pending(): Deferred<Engine> =
    synchronized(lock) {
      val current = loading
      // A failed attempt counts as cancelled: it is replaced.
      if (current != null && !current.isCancelled) {
        current
      } else {
        scope.async { open() }.also { loading = it }
      }
    }

  /** On the native thread. */
  @OptIn(ExperimentalApi::class)
  private fun open(): Engine {
    check(!closed) { "the model is closed" }
    unusable?.let { throw IllegalStateException("the model is unusable: $it") }
    _state.value = EngineState.Preparing
    try {
      memory.require()
      check(cacheDir.isDirectory || cacheDir.mkdirs()) { "Cannot create $cacheDir" }
      ExperimentalFlags.enableBenchmark = true
      val t0 = System.nanoTime()
      val e =
        Engine(
          EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.GPU(),
            visionBackend = Backend.CPU(),
            maxNumImages = 1,
            cacheDir = cacheDir.absolutePath,
          )
        )
      e.initialize()
      engine = e
      engineMs = ms(System.nanoTime() - t0)
      Log.i(TAG, "ENGINE_READY engine_ms=$engineMs")
      if (!stepping.get()) _state.value = EngineState.Ready(engineMs)
      return e
    } catch (t: Throwable) {
      Log.e(TAG, "ENGINE_FAILED ${t.javaClass.simpleName}: ${t.message}")
      _state.value = EngineState.Failed(t.message ?: t.javaClass.simpleName)
      throw t
    }
  }

  /**
   * One inference for [request] on [picture] (PNG bytes, 1920 x 1088): the agent's first step, or with
   * [firstReply] (the first step's reply, a call for a screenshot) its second, the picture being the
   * answer to that call. Streams the reply. One at a time. [renderPrompt] (debug) first emits the
   * prompt as the chat template rendered it.
   */
  @OptIn(ExperimentalApi::class)
  fun step(
    picture: ByteArray,
    request: String,
    firstReply: String?,
    maxOutputTokens: Int,
    renderPrompt: Boolean = false,
  ): Flow<StepEvent> = flow {
    check(!closed) { "the model is closed" }
    unusable?.let { throw IllegalStateException("the model is unusable: $it") }
    check(stepping.compareAndSet(false, true)) { "another step is running" }
    var live: Conversation? = null
    try {
      val asked = System.nanoTime()
      val pending = pending()
      if (!pending.isCompleted) emit(StepEvent.Waiting)
      val made = pending.await()
      _state.value = EngineState.Deciding
      val t1 = System.nanoTime()
      val conversation =
        withContext(native) {
          made.createConversation(
            ConversationConfig(
              systemInstruction = Contents.of(systemPrompt),
              initialMessages =
                if (firstReply == null) {
                  emptyList()
                } else {
                  listOf(Message.user(Step.requestTurn(request)), Message.model(firstReply))
                },
            )
          )
        }
      val conversationMs = ms(System.nanoTime() - t1)
      live = conversation
      val message =
        if (firstReply == null) {
          Message.user(Contents.of(Content.ImageBytes(picture), Content.Text(Step.stepText(request))))
        } else {
          Message.user(
            Contents.of(
              Content.Text(Step.RESULT_BEFORE),
              Content.ImageBytes(picture),
              Content.Text(Step.RESULT_AFTER),
            )
          )
        }
      if (renderPrompt) {
        val rendered =
          withContext(native) {
            val preface = runCatching { conversation.renderPrefaceIntoString() }.getOrElse { "" }
            preface +
              runCatching { conversation.renderMessageIntoString(message, emptyMap()) }
                .getOrElse { "render failed: ${it.javaClass.simpleName}: ${it.message}" }
          }
        emit(StepEvent.Rendered(rendered))
      }
      emit(StepEvent.Started(ms(System.nanoTime() - asked), engineMs, conversationMs))
      val events = Channel<StepEvent>(Channel.UNLIMITED)
      val done = CompletableDeferred<Unit>()
      val callback =
        object : MessageCallback {
          override fun onMessage(message: Message) {
            val text =
              message.contents.contents.joinToString("") { (it as? Content.Text)?.text ?: "" }
            if (text.isNotEmpty()) events.trySend(StepEvent.Text(text))
          }

          override fun onDone() {
            done.complete(Unit)
            events.close()
          }

          override fun onError(throwable: Throwable) {
            done.complete(Unit)
            if (throwable is JavaCancellationException || throwable is CancellationException) {
              events.close()
            } else {
              events.close(throwable)
            }
          }
        }
      inFlight = conversation
      if (stopAsked) {
        // Stop came before this inference started: nothing to send.
        done.complete(Unit)
        events.close()
      } else {
        try {
          conversation.sendMessageAsync(
            message,
            callback,
            emptyMap(),
            maxOutputToken = maxOutputTokens,
          )
        } catch (t: Throwable) {
          done.complete(Unit)
          throw t
        }
      }
      try {
        for (e in events) emit(e)
      } finally {
        withContext(NonCancellable) {
          if (!done.isCompleted) runCatching { conversation.cancelProcess() }
          if (withTimeoutOrNull(NATIVE_STOP_MS) { done.await() } == null) {
            unusable = "a step did not stop within $NATIVE_STOP_MS ms"
            Log.e(TAG, "$unusable; its Conversation is not released")
            _state.value = EngineState.Failed(unusable!!)
            live = null
          }
          inFlight = null
        }
      }
      emit(StepEvent.Finished(withContext(native) { stagesOf(conversation) }))
    } finally {
      withContext(NonCancellable) {
        val used = live
        if (used != null) {
          withContext(native) {
            runCatching { used.close() }.onFailure { Log.w(TAG, "conversation.close: ${it.message}") }
          }
        }
        stepping.set(false)
        if (!closed && unusable == null && engine != null) _state.value = EngineState.Ready(engineMs)
      }
    }
  }

  @OptIn(ExperimentalApi::class)
  private fun stagesOf(conversation: Conversation): Stages? =
    runCatching {
        val b = conversation.getBenchmarkInfo()
        Stages(
          timeToFirstTokenS = b.timeToFirstTokenInSecond,
          prefillTokens = b.lastPrefillTokenCount,
          prefillTokensPerS = b.lastPrefillTokensPerSecond,
          decodeTokens = b.lastDecodeTokenCount,
          decodeTokensPerS = b.lastDecodeTokensPerSecond,
          marks = b.markDurationsInSecond,
        )
      }
      .onFailure { Log.w(TAG, "benchmark info: ${it.message}") }
      .getOrNull()

  /** A new decision: a Stop from the last one no longer counts. */
  fun clearStop() {
    stopAsked = false
  }

  /**
   * Stops the inference in flight and any that has not started; its flow ends without an error, once
   * the runtime stops.
   */
  fun cancel() {
    stopAsked = true
    inFlight?.let { runCatching { it.cancelProcess() } }
  }

  /** Releases the Conversation and the Engine; a step in flight is stopped first. */
  suspend fun close() {
    closed = true
    cancel()
    withContext(NonCancellable) {
      withTimeoutOrNull(NATIVE_STOP_MS + 1_000) {
        while (stepping.get()) kotlinx.coroutines.delay(50)
      }
      val pending = synchronized(lock) { loading.also { loading = null } }
      if (pending != null) runCatching { pending.await() }
      withContext(native) {
        runCatching { engine?.close() }.onFailure { Log.w(TAG, "engine.close: ${it.message}") }
        engine = null
      }
    }
    nativeThread.shutdown()
  }

  companion object {
    private const val TAG = "ClickOrCode"
    private const val THREAD = "click-or-code-litertlm"
    private const val NATIVE_STOP_MS = 10_000L

    private fun ms(nanos: Long) = nanos / 1_000_000
  }
}
