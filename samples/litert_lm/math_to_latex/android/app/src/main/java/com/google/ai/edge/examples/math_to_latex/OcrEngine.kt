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

package com.google.ai.edge.examples.math_to_latex

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
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

/** Where the reader is between reads. */
sealed class ReaderState {
  data object Idle : ReaderState()

  /** An Engine is being made (the first one loads the model and fills the runtime's caches). */
  data object Preparing : ReaderState()

  /** An Engine and its Conversation wait for the next read. */
  data class Ready(val engineMs: Long, val conversationMs: Long) : ReaderState()

  data object Reading : ReaderState()

  /** The last attempt to make an Engine failed; the next read tries again. */
  data class Failed(val message: String) : ReaderState()
}

/** What a read reports while it runs, in order. */
sealed class ReadEvent {
  /** The next Engine is still being made; the read starts when it is ready. */
  data object Waiting : ReadEvent()

  /** The prompt as the chat template rendered it (debug reads only). */
  data class Rendered(val prompt: String) : ReadEvent()

  /** The message went in, [waitMs] after the read was asked for. */
  data class Started(val waitMs: Long, val engineMs: Long, val conversationMs: Long) : ReadEvent()

  /** One streamed piece of the model's answer. */
  data class Text(val text: String) : ReadEvent()
}

/**
 * MiniCPM-OCR-1.0 on LiteRT-LM (text decoder and vision encoder on the CPU), one Engine per read.
 *
 * The text decoder carries a recurrent state, and a second conversation on the same Engine would
 * continue from the first one's state (LiteRT-LM issue #3165), so every read gets a new Engine and
 * Conversation, as the model card says. To keep that out of the wait, the next Engine is made as
 * soon as a read ends, while the user frames the next region: a read then only pays for the vision
 * encoder, the prompt and the answer. Engines are made and released on one thread, one at a time,
 * so two never hold memory together.
 */
class OcrEngine(
  private val modelFile: File,
  private val cacheDir: File,
  private val memory: MemoryCheck = MemoryCheck(),
) {
  private class Slot(
    val engine: Engine,
    val conversation: Conversation,
    val engineMs: Long,
    val conversationMs: Long,
  )

  private val nativeThread =
    Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }
  private val native = nativeThread.asCoroutineDispatcher()
  private val scope = CoroutineScope(SupervisorJob() + native)
  private val reading = AtomicBoolean(false)
  private val lock = Any()
  private var next: Deferred<Slot>? = null
  private val _state = MutableStateFlow<ReaderState>(ReaderState.Idle)
  @Volatile private var inFlight: Conversation? = null
  @Volatile private var closed = false
  @Volatile private var unusable: String? = null

  val state: StateFlow<ReaderState> = _state.asStateFlow()

  /**
   * Starts making the next Engine and Conversation unless one is ready or being made. A failed
   * attempt is forgotten, so the next call tries again.
   */
  fun prepareNext() {
    pending()
  }

  private fun pending(): Deferred<Slot> =
    synchronized(lock) {
      val current = next
      // A failed attempt counts as cancelled: it is replaced.
      if (current != null && !current.isCancelled) {
        current
      } else {
        scope.async { open() }.also { next = it }
      }
    }

  /** On the native thread. */
  private fun open(): Slot {
    check(!closed) { "the reader is closed" }
    unusable?.let { throw IllegalStateException("the reader is unusable: $it") }
    _state.value = ReaderState.Preparing
    try {
      memory.require()
      check(cacheDir.isDirectory || cacheDir.mkdirs()) { "Cannot create $cacheDir" }
      val t0 = System.nanoTime()
      val engine =
        Engine(
          EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.CPU(),
            visionBackend = Backend.CPU(),
            maxNumImages = Pictures.MAX_PICTURES,
            cacheDir = cacheDir.absolutePath,
          )
        )
      engine.initialize()
      val t1 = System.nanoTime()
      val conversation =
        try {
          engine.createConversation(ConversationConfig())
        } catch (t: Throwable) {
          runCatching { engine.close() }
          throw t
        }
      val slot = Slot(engine, conversation, ms(t1 - t0), ms(System.nanoTime() - t1))
      Log.i(TAG, "ENGINE_READY engine_ms=${slot.engineMs} conversation_ms=${slot.conversationMs}")
      if (!reading.get()) _state.value = ReaderState.Ready(slot.engineMs, slot.conversationMs)
      return slot
    } catch (t: Throwable) {
      Log.e(TAG, "ENGINE_FAILED ${t.javaClass.simpleName}: ${t.message}")
      _state.value = ReaderState.Failed(t.message ?: t.javaClass.simpleName)
      throw t
    }
  }

  /** On the native thread. */
  private fun release(slot: Slot) {
    runCatching { slot.conversation.close() }
      .onFailure { Log.w(TAG, "conversation.close: ${it.message}") }
    runCatching { slot.engine.close() }.onFailure { Log.w(TAG, "engine.close: ${it.message}") }
  }

  /**
   * Reads [pictures] (PNG bytes, 448 x 448) with [prompt] in one user message, [extraContext] going
   * to the chat template, and streams the answer. The Engine is released when the answer ends (or
   * the collector stops), and the next one is started. One read at a time. [renderPrompt] (debug)
   * first emits the prompt as the chat template rendered it.
   */
  @OptIn(ExperimentalApi::class)
  fun read(
    pictures: List<ByteArray>,
    prompt: String,
    extraContext: Map<String, Any>,
    maxOutputTokens: Int,
    renderPrompt: Boolean = false,
  ): Flow<ReadEvent> = flow {
    check(!closed) { "the reader is closed" }
    unusable?.let { throw IllegalStateException("the reader is unusable: $it") }
    check(reading.compareAndSet(false, true)) { "another read is running" }
    var slot: Slot? = null
    try {
      val asked = System.nanoTime()
      val pending = pending()
      if (!pending.isCompleted) emit(ReadEvent.Waiting)
      val taken =
        try {
          pending.await()
        } finally {
          synchronized(lock) { if (next === pending) next = null }
        }
      slot = taken
      _state.value = ReaderState.Reading
      val message =
        Message.user(Contents.of(pictures.map { Content.ImageBytes(it) } + Content.Text(prompt)))
      if (renderPrompt) {
        val rendered =
          withContext(native) {
            runCatching { taken.conversation.renderMessageIntoString(message, extraContext) }
              .getOrElse { "render failed: ${it.javaClass.simpleName}: ${it.message}" }
          }
        emit(ReadEvent.Rendered(rendered))
      }
      emit(ReadEvent.Started(ms(System.nanoTime() - asked), taken.engineMs, taken.conversationMs))
      val events = Channel<ReadEvent>(Channel.UNLIMITED)
      val done = CompletableDeferred<Unit>()
      val callback =
        object : MessageCallback {
          override fun onMessage(message: Message) {
            val text =
              message.contents.contents.joinToString("") { (it as? Content.Text)?.text ?: "" }
            if (text.isNotEmpty()) events.trySend(ReadEvent.Text(text))
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
      inFlight = taken.conversation
      try {
        taken.conversation.sendMessageAsync(
          message,
          callback,
          extraContext,
          maxOutputToken = maxOutputTokens,
        )
      } catch (t: Throwable) {
        done.complete(Unit)
        throw t
      }
      try {
        for (e in events) emit(e)
      } finally {
        withContext(NonCancellable) {
          if (!done.isCompleted) runCatching { taken.conversation.cancelProcess() }
          if (withTimeoutOrNull(NATIVE_STOP_MS) { done.await() } == null) {
            unusable = "a read did not stop within $NATIVE_STOP_MS ms"
            Log.e(TAG, "$unusable; its Engine is not released")
            _state.value = ReaderState.Failed(unusable!!)
            slot = null
          }
          inFlight = null
        }
      }
    } finally {
      withContext(NonCancellable) {
        val used = slot
        if (used != null) withContext(native) { release(used) }
        reading.set(false)
        if (!closed && unusable == null && used != null) {
          prepareNext()
        } else if (!closed && unusable == null && _state.value !is ReaderState.Failed) {
          _state.value = ReaderState.Idle
        }
      }
    }
  }

  /**
   * Stops the read in flight; its flow ends without an error. It does not end at once: on LiteRT-LM
   * 0.18.1 a Stop 1.5 s into a whole page (9 pictures) on a Galaxy S26 ended the read 24.6 s after
   * it was asked for (23 s after the Stop), with no text.
   */
  fun cancel() {
    inFlight?.let { runCatching { it.cancelProcess() } }
  }

  /** Releases the prepared Engine; a read in flight is stopped first. */
  suspend fun close() {
    closed = true
    cancel()
    withContext(NonCancellable) {
      // A read in flight releases its own Engine on the native thread first.
      withTimeoutOrNull(NATIVE_STOP_MS + 1_000) {
        while (reading.get()) kotlinx.coroutines.delay(50)
      }
      val pending = synchronized(lock) { next.also { next = null } }
      if (pending != null) {
        val slot = runCatching { pending.await() }.getOrNull()
        if (slot != null) withContext(native) { release(slot) }
      }
    }
    nativeThread.shutdown()
  }

  companion object {
    private const val TAG = "MathToLatex"
    private const val THREAD = "math-to-latex-litertlm"
    private const val NATIVE_STOP_MS = 10_000L

    private fun ms(nanos: Long) = nanos / 1_000_000
  }
}
