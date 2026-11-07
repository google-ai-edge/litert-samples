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

package com.google.ai.edge.examples.voice_to_text

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.concurrent.CancellationException as JavaCancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random
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

/** The Engine could not be made, so a recording could not go to the model; the message says why. */
class EngineUnavailableException(message: String) : IllegalStateException(message)

/** Where the model is, from a downloaded file to an Engine that takes recordings. */
sealed class EngineState {
  data object Idle : EngineState()

  /** The Engine is being made: the model loads and the GPU programs are built or read back. */
  data object Loading : EngineState()

  /** A first short message runs through the new Engine so the first recording is not slower. */
  data object WarmingUp : EngineState()

  data class Ready(val engineMs: Long, val warmUpMs: Long) : EngineState()

  /** Making the Engine failed; [message] says why. Load again to retry. */
  data class Failed(val message: String) : EngineState()
}

/** What a transcription reports while it runs, in order. */
sealed class SpeechEvent {
  /** The Engine is still being made; the transcription starts when it is ready. */
  data object Waiting : SpeechEvent()

  /** The message went in, [waitMs] after the transcription was asked for. */
  data class Started(val waitMs: Long) : SpeechEvent()

  /** One streamed piece of the reply. */
  data class Text(val text: String) : SpeechEvent()

  /** The reply ended; [tokens] = prompt and reply in the context. */
  data class Done(val tokens: Int?) : SpeechEvent()

  /** Stop was tapped; the reply ended early. */
  data object Cancelled : SpeechEvent()
}

/**
 * granite-speech-5.0-2b on LiteRT-LM: one Engine with the language model and the audio encoder on
 * the GPU, and a new Conversation for every recording, as the model card measures it (a second
 * message in one conversation would start from the previous answer). Native calls run on one
 * thread, one at a time.
 */
class SpeechEngine(
  private val modelFile: File,
  private val cacheDir: File,
  private val memory: MemoryCheck = MemoryCheck(),
) {
  private val nativeThread =
    Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }
  private val native = nativeThread.asCoroutineDispatcher()
  private val scope = CoroutineScope(SupervisorJob() + native)
  private val busy = AtomicBoolean(false)
  private val lock = Any()
  private var loading: Deferred<Engine>? = null
  private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
  @Volatile private var inFlight: Conversation? = null
  @Volatile private var closed = false
  @Volatile private var unusable: String? = null

  val state: StateFlow<EngineState> = _state.asStateFlow()

  /** Starts making the Engine unless it is made or being made. A failed attempt is retried. */
  fun load() {
    engine()
  }

  private fun engine(): Deferred<Engine> =
    synchronized(lock) {
      val current = loading
      if (current != null && !current.isCancelled) current
      else scope.async { open() }.also { loading = it }
    }

  /** On the native thread. */
  private fun open(): Engine {
    check(!closed) { "the engine is closed" }
    _state.value = EngineState.Loading
    try {
      memory.require()
      check(cacheDir.isDirectory || cacheDir.mkdirs()) { "Cannot create $cacheDir" }
      val t0 = System.nanoTime()
      val engine =
        Engine(
          EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.GPU(),
            audioBackend = Backend.GPU(),
            cacheDir = cacheDir.absolutePath,
          )
        )
      engine.initialize()
      val engineMs = ms(System.nanoTime() - t0)
      _state.value = EngineState.WarmingUp
      val t1 = System.nanoTime()
      try {
        warmUp(engine)
      } catch (t: Throwable) {
        runCatching { engine.close() }
        throw t
      }
      val warmUpMs = ms(System.nanoTime() - t1)
      Log.i(TAG, "ENGINE_READY engine_ms=$engineMs warm_up_ms=$warmUpMs")
      _state.value = EngineState.Ready(engineMs, warmUpMs)
      return engine
    } catch (t: Throwable) {
      Log.e(TAG, "ENGINE_FAILED ${t.javaClass.simpleName}: ${t.message}")
      _state.value = EngineState.Failed(Messages.engineFailed(t))
      throw t
    }
  }

  /**
   * One short message (two seconds of faint noise, a reply of at most 4 tokens) before the first
   * recording. On the runtime's first conversation, part of the work runs slower; the warm-up takes
   * that cost while the screen still says "Loading".
   */
  private fun warmUp(engine: Engine) {
    val random = Random(7)
    val noise = FloatArray(2 * Wav.SAMPLE_RATE) { (random.nextFloat() - 0.5f) * 0.002f }
    engine.createConversation(conversationConfig()).use { conversation ->
      conversation.sendMessage(
        Contents.of(Content.AudioBytes(Wav.encodeFloat(noise)), Content.Text(Prompts.prompt(Task.TRANSCRIPT, emptyList()))),
        maxOutputToken = 4,
      )
    }
  }

  private fun conversationConfig() =
    // Greedy decoding, as the model card's checks run.
    ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 1.0))

  /**
   * Sends [wav] (16 kHz mono) and [prompt] as one user message in a new Conversation and streams
   * the reply. One transcription at a time; the Conversation is closed when the reply ends or the
   * collector stops.
   */
  fun transcribe(wav: ByteArray, prompt: String, maxOutputTokens: Int): Flow<SpeechEvent> = flow {
    check(!closed) { "the engine is closed" }
    unusable?.let { throw IllegalStateException(it) }
    check(busy.compareAndSet(false, true)) { "another transcription is running" }
    var conversation: Conversation? = null
    try {
      val asked = System.nanoTime()
      val pending = engine()
      if (!pending.isCompleted) emit(SpeechEvent.Waiting)
      val engine =
        try {
          pending.await()
        } catch (e: CancellationException) {
          throw e
        } catch (t: Throwable) {
          throw EngineUnavailableException(Messages.engineFailed(t))
        }
      val made = withContext(native) { engine.createConversation(conversationConfig()) }
      conversation = made
      emit(SpeechEvent.Started(ms(System.nanoTime() - asked)))
      val events = Channel<SpeechEvent>(Channel.UNLIMITED)
      val done = CompletableDeferred<Unit>()
      val callback =
        object : MessageCallback {
          override fun onMessage(message: Message) {
            val text =
              message.contents.contents.joinToString("") { (it as? Content.Text)?.text ?: "" }
            if (text.isNotEmpty()) events.trySend(SpeechEvent.Text(text))
          }

          override fun onDone() {
            done.complete(Unit)
            events.close()
          }

          override fun onError(throwable: Throwable) {
            done.complete(Unit)
            if (throwable is JavaCancellationException || throwable is CancellationException) {
              events.trySend(SpeechEvent.Cancelled)
              events.close()
            } else {
              events.close(throwable)
            }
          }
        }
      inFlight = made
      try {
        withContext(native) {
          made.sendMessageAsync(
            Contents.of(Content.AudioBytes(wav), Content.Text(prompt)),
            callback,
            maxOutputToken = maxOutputTokens,
          )
        }
      } catch (t: Throwable) {
        done.complete(Unit)
        throw t
      }
      var cancelled = false
      try {
        for (e in events) {
          if (e is SpeechEvent.Cancelled) cancelled = true
          emit(e)
        }
      } finally {
        withContext(NonCancellable) {
          if (!done.isCompleted) runCatching { made.cancelProcess() }
          if (withTimeoutOrNull(NATIVE_STOP_MS) { done.await() } == null) {
            unusable = "A transcription did not stop within ${NATIVE_STOP_MS / 1000} s. Close and reopen the app."
            Log.e(TAG, "ENGINE_STUCK; the conversation is not released")
            _state.value = EngineState.Failed(unusable!!)
            conversation = null
          }
          inFlight = null
        }
      }
      if (!cancelled) {
        val tokens = withContext(native) { runCatching { made.getTokenCount() }.getOrNull() }
        emit(SpeechEvent.Done(tokens))
      }
    } finally {
      withContext(NonCancellable) {
        val used = conversation
        if (used != null) {
          withContext(native) {
            runCatching { used.close() }.onFailure { Log.w(TAG, "conversation.close: ${it.message}") }
          }
        }
        busy.set(false)
      }
    }
  }

  /** Stops the transcription in flight; its flow ends with [SpeechEvent.Cancelled]. */
  fun cancel() {
    inFlight?.let { runCatching { it.cancelProcess() } }
  }

  /** Releases the Engine; a transcription in flight is stopped first. */
  suspend fun close() {
    closed = true
    cancel()
    withContext(NonCancellable) {
      withTimeoutOrNull(NATIVE_STOP_MS + 1_000) {
        while (busy.get()) kotlinx.coroutines.delay(50)
      }
      val pending = synchronized(lock) { loading.also { loading = null } }
      if (pending != null) {
        val engine = runCatching { pending.await() }.getOrNull()
        if (engine != null) {
          withContext(native) {
            runCatching { engine.close() }.onFailure { Log.w(TAG, "engine.close: ${it.message}") }
          }
        }
      }
    }
    nativeThread.shutdown()
  }

  companion object {
    private const val TAG = "VoiceToText"
    private const val THREAD = "voice-to-text-litertlm"
    private const val NATIVE_STOP_MS = 10_000L

    private fun ms(nanos: Long) = nanos / 1_000_000
  }
}
