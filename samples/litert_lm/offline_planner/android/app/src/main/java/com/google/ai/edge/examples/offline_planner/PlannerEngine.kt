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

package com.google.ai.edge.examples.offline_planner

import android.os.SystemClock
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
import com.google.ai.edge.litertlm.ThinkingConfig
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * LFM3-1.7B on LiteRT-LM, with one new [Engine] for every request.
 *
 * LFM3 keeps a running short-convolution state, and on LiteRT-LM a second conversation on the
 * same Engine starts from the state the first one left (LiteRT-LM issue #3165). So each request
 * gets an Engine of its own: after a reply the app closes that Engine and creates the next one in
 * the background, and the next tap finds it ready. All native calls run on one thread.
 */
class PlannerEngine(private val modelPath: String, private val cacheDir: String, val useGpu: Boolean) {

  /** Callbacks on the runtime's or the engine's thread; the screen posts them to the main thread. */
  interface Listener {
    /** The reply so far. */
    fun onText(text: String)

    fun onDone(reply: Reply)

    fun onError(message: String)
  }

  /** A finished reply. Times are `SystemClock.elapsedRealtimeNanos()`; [kvTokens] is the
   * conversation's token count after the reply (prompt and reply). */
  data class Reply(
    val text: String,
    val thought: String,
    val firstTokenNs: Long,
    val doneNs: Long,
    val kvTokens: Int,
    val chunks: Int,
  )

  private val executor =
    Executors.newSingleThreadExecutor { r -> Thread(r, "planner-litertlm").apply { isDaemon = true } }
  private var engine: Engine? = null

  @Volatile
  var ready = false
    private set

  /** Milliseconds the last Engine took to initialize. */
  @Volatile
  var lastInitMs = -1L
    private set

  val backendName: String
    get() = if (useGpu) "GPU" else "CPU"

  /** Closes the previous Engine, if any, and initializes a new one. */
  fun prepare(onResult: (Result<Long>) -> Unit) {
    ready = false
    executor.execute {
      closeEngine()
      val t0 = SystemClock.elapsedRealtime()
      val result = runCatching {
        val e =
          Engine(
            EngineConfig(
              modelPath = modelPath,
              backend = if (useGpu) Backend.GPU() else Backend.CPU(),
              cacheDir = cacheDir,
            )
          )
        try {
          e.initialize()
        } catch (t: Throwable) {
          runCatching { e.close() }
          throw t
        }
        engine = e
        SystemClock.elapsedRealtime() - t0
      }
      result
        .onSuccess {
          lastInitMs = it
          ready = true
          Log.i(TAG, "engine ready on $backendName in $it ms")
        }
        .onFailure { Log.e(TAG, "engine initialize failed on $backendName", it) }
      onResult(result)
    }
  }

  /**
   * Sends [request] with [system] and the tools in a new conversation on the prepared Engine:
   * greedy decoding, thinking off, at most [MAX_OUTPUT_TOKENS] tokens. The Engine is closed
   * afterwards, whatever happened; call [prepare] before the next request.
   */
  fun ask(system: String, request: String, listener: Listener) {
    ready = false
    executor.execute {
      val e = engine
      if (e == null) {
        listener.onError("The model is not loaded.")
        return@execute
      }
      var conversation: Conversation? = null
      try {
        conversation =
          e.createConversation(
            ConversationConfig(
              systemInstruction = Contents.of(system),
              samplerConfig = GREEDY,
              automaticToolCalling = false,
              extraContext = mapOf("tools" to Planner.toolsJson()),
              maxOutputToken = MAX_OUTPUT_TOKENS,
              thinkingConfig = NO_THINKING,
            )
          )
        val text = StringBuilder()
        val thought = StringBuilder()
        var firstNs = 0L
        var chunks = 0
        var error: Throwable? = null
        val done = CountDownLatch(1)
        conversation.sendMessageAsync(
          Message.user(request),
          object : MessageCallback {
            override fun onMessage(message: Message) {
              if (firstNs == 0L) firstNs = SystemClock.elapsedRealtimeNanos()
              chunks++
              for (c in message.contents.contents) if (c is Content.Text) text.append(c.text)
              for (v in message.channels.values) thought.append(v)
              listener.onText(text.toString())
            }

            override fun onDone() {
              done.countDown()
            }

            override fun onError(throwable: Throwable) {
              error = throwable
              done.countDown()
            }
          },
          maxOutputToken = MAX_OUTPUT_TOKENS,
          thinkingConfig = NO_THINKING,
        )
        if (!done.await(REPLY_TIMEOUT_S, TimeUnit.SECONDS)) {
          conversation.cancelProcess()
          done.await(STOP_TIMEOUT_S, TimeUnit.SECONDS)
          error = error ?: IllegalStateException("no reply within $REPLY_TIMEOUT_S s")
        }
        val doneNs = SystemClock.elapsedRealtimeNanos()
        val failure = error
        if (failure != null) {
          listener.onError("Generation failed: ${failure.message}")
        } else {
          val kv = runCatching { conversation.getTokenCount() }.getOrDefault(-1)
          listener.onDone(Reply(text.toString(), thought.toString(), firstNs, doneNs, kv, chunks))
        }
      } catch (t: Throwable) {
        Log.e(TAG, "request failed", t)
        listener.onError("${t.javaClass.simpleName}: ${t.message}")
      } finally {
        runCatching { conversation?.close() }
        closeEngine()
      }
    }
  }

  /** Releases the Engine (after any request in flight has finished). */
  fun close() {
    executor.execute { closeEngine() }
    executor.shutdown()
  }

  private fun closeEngine() {
    ready = false
    engine?.let { runCatching { it.close() }.onFailure { t -> Log.w(TAG, "engine close: ${t.message}") } }
    engine = null
  }

  companion object {
    private const val TAG = "OfflinePlanner"

    /** Enough for three calls with a short message each. */
    const val MAX_OUTPUT_TOKENS = 256
    private const val REPLY_TIMEOUT_S = 120L
    private const val STOP_TIMEOUT_S = 10L

    /** Greedy decoding: the same request on the same phone gives the same calls. */
    private val GREEDY = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0)

    /** The model's chat template then opens the answer with an empty thought (`<think></think>`). */
    private val NO_THINKING = ThinkingConfig(enableThinking = false)
  }
}
