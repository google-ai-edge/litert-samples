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

package com.google.ai.edge.examples.where_to_put

import android.app.ActivityManager
import android.content.Context
import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import java.io.File
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** One streamed piece of the model's reply: the answer's text and, with thinking on, the thought's. */
data class Delta(val text: String, val thought: String)

/**
 * The LiteRT-LM engine for LFM3-VL-2B. One engine stays loaded for the life of the process; every request gets a
 * conversation of its own.
 */
object ModelEngine {
  const val FILE_NAME = "LFM3-VL-2B_int4_ctx4096.litertlm"
  const val FILE_BYTES = 2_071_674_880L

  /**
   * Thinking. On: the model first writes its reasoning (hundreds of tokens, in the bundle's "thought" channel), then
   * the point. Off: the point comes as the first words of the answer, in about 45 tokens. The chat template's
   * `enable_thinking` turns it on or off.
   */
  const val THINKING = true

  /** The file's context: the picture and the request take about 340 of its 4,096 tokens, the reply the rest. */
  const val CONTEXT_TOKENS = 4_096

  /**
   * The reply's cap. With thinking on it is the whole context, so the reply can use all the room the picture and the
   * request leave; LiteRT-LM stops it at the context's end and ends it like a finished reply. With thinking off, 256.
   */
  val MAX_OUTPUT_TOKENS = if (THINKING) CONTEXT_TOKENS else 256

  /** The cap the next conversations use; the measuring tool can lower it (MainActivity). */
  @Volatile var maxOutputTokens = MAX_OUTPUT_TOKENS

  // On a Pixel 11 Pro (PowerVR GPU) this app took 6.0 GB while it answered, most of it GPU memory; with less memory
  // available than that, the app does not start the load.
  const val NEEDED_BYTES = 6_000_000_000L

  /** The memory the check before the load asks for; the measuring tool can raise it (MainActivity). */
  @Volatile var neededBytes = NEEDED_BYTES

  /** The backends of the text decoder and of the vision encoder ("gpu" or "cpu"); the measuring tool can change them. */
  @Volatile var textBackend = "gpu"
  @Volatile var visionBackend = "gpu"

  sealed class State {
    object NotLoaded : State()
    object Loading : State()
    data class Ready(val loadMs: Long, val backends: String) : State()
    /** [retry]: the cause (no file, an incomplete file, too little memory) can change while the app is away. */
    data class Unavailable(val message: String, val retry: Boolean = false) : State()
  }

  @Volatile var state: State = State.NotLoaded
    private set

  private var engine: Engine? = null
  private val lock = Mutex()

  fun modelFile(context: Context): File = File(context.getExternalFilesDir(null), FILE_NAME)

  /** "GPU" for the text decoder, "vision GPU" for the encoder, as the status line shows them. */
  fun backendsLabel(): String = "${textBackend.uppercase(Locale.US)}, vision ${visionBackend.uppercase(Locale.US)}"

  /** Checks the file and the free memory, then loads the engine (once; a failed check can be tried again). */
  suspend fun load(context: Context): State = lock.withLock {
    (state as? State.Ready)?.let { return it }
    val file = modelFile(context)
    if (!file.isFile) {
      return unavailable(
        "The model file is not on this phone yet. Missing: $FILE_NAME (${gb(FILE_BYTES)}). " +
          "Push it into ${file.parent} with adb (see the README), then reopen the app.",
        retry = true,
      )
    }
    // The file's own header says how long a whole copy is, whichever form the file has: its weights
    // inside the graphs, or in TFLiteWeights sections.
    val wholeBytes = LitertlmHeader.read(file)?.fileBytes
    if (wholeBytes == null) {
      return unavailable(
        "$FILE_NAME does not begin the way a .litertlm file does. The copy may be incomplete: push it again.",
        retry = true,
      )
    }
    if (file.length() != wholeBytes) {
      return unavailable(
        "$FILE_NAME is ${file.length()} bytes, but its header lists $wholeBytes bytes. " +
          "The copy may be incomplete: push it again.",
        retry = true,
      )
    }
    val memory = ActivityManager.MemoryInfo()
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
    if (memory.availMem < neededBytes) {
      return unavailable(
        "This phone has ${gb(memory.availMem)} of memory available and the model needs about " +
          "${gb(neededBytes)}. Close other apps and reopen this one. Nothing was loaded.",
        retry = true,
      )
    }
    state = State.Loading
    val start = SystemClock.elapsedRealtime()
    return try {
      val e = withContext(Dispatchers.IO) {
        Engine(
          EngineConfig(
            modelPath = file.absolutePath,
            backend = backend(textBackend),
            visionBackend = backend(visionBackend),
            maxNumImages = 1,
            cacheDir = context.cacheDir.absolutePath,
          )
        ).also { it.initialize() }
      }
      engine = e
      State.Ready(SystemClock.elapsedRealtime() - start, backendsLabel()).also { state = it }
    } catch (t: Throwable) {
      unavailable("The model could not be loaded (${backendsLabel()}): ${t.message ?: t.javaClass.simpleName}")
    }
  }

  /**
   * Sends the picture and the request in a new conversation and streams the reply. The thought, when thinking is on,
   * comes in the bundle's "thought" channel, apart from the answer; a reply that reaches the cap or the context's end
   * simply ends, and then has no answer if the model was still thinking.
   */
  fun ask(png: ByteArray, instruction: String): Flow<Delta> {
    val e = engine ?: error("The model is not loaded.")
    val thinking = mapOf<String, Any>("enable_thinking" to THINKING)
    val conversation = e.createConversation(
      ConversationConfig(extraContext = thinking, maxOutputToken = maxOutputTokens)
    )
    return conversation
      .sendMessageAsync(Contents.of(Content.ImageBytes(png), Content.Text(instruction)), thinking)
      .map { message ->
        Delta(
          text = message.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text },
          thought = message.channels["thought"] ?: "",
        )
      }
      .onCompletion { cause ->
        // The screen went away mid-reply: stop the reply before the conversation is freed.
        if (cause != null) runCatching { conversation.cancelProcess() }
        conversation.close()
      }
      .flowOn(Dispatchers.IO)
  }

  private fun backend(name: String): Backend = if (name == "cpu") Backend.CPU() else Backend.GPU()

  private fun unavailable(message: String, retry: Boolean = false): State =
    State.Unavailable(message, retry).also { state = it }

  private fun gb(bytes: Long) = String.format(Locale.US, "%.1f GB", bytes / 1e9)
}
