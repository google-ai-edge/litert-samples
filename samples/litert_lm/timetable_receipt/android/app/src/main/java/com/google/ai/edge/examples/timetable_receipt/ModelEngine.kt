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

package com.google.ai.edge.examples.timetable_receipt

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

/**
 * The LiteRT-LM engine for LFM3-VL-2B. One engine stays loaded for the life of the process (a
 * GPU load takes tens of seconds on a phone); every question gets a conversation of its own.
 */
object ModelEngine {
  const val FILE_NAME = "LFM3-VL-2B_f16.litertlm"
  const val FILE_BYTES = 4_439_967_995L

  // On a Galaxy S26 this app's resident memory peaked at 4.8 GB while it answered about pictures on
  // the GPU; with less memory available than that, the app does not start the load.
  const val NEEDED_BYTES = 4_800_000_000L
  private const val MAX_OUTPUT_TOKENS = 256

  /** The memory the check before the load asks for; the measuring tool can raise it (MainActivity). */
  @Volatile var neededBytes = NEEDED_BYTES

  sealed class State {
    object NotLoaded : State()
    object Loading : State()
    data class Ready(val loadMs: Long) : State()
    /** [retry]: the cause (no file, an incomplete file, too little memory) can change while the app is away. */
    data class Unavailable(val message: String, val retry: Boolean = false) : State()
  }

  @Volatile var state: State = State.NotLoaded
    private set

  private var engine: Engine? = null
  private val lock = Mutex()

  fun modelFile(context: Context): File = File(context.getExternalFilesDir(null), FILE_NAME)

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
    if (file.length() != FILE_BYTES) {
      return unavailable(
        "$FILE_NAME is ${file.length()} bytes, but the published file has $FILE_BYTES bytes. " +
          "The copy may be incomplete: push it again.",
        retry = true,
      )
    }
    val memory = ActivityManager.MemoryInfo()
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
    if (memory.availMem < neededBytes) {
      return unavailable(
        "This phone has ${gb(memory.availMem)} of memory available and the model needs about " +
          "${gb(neededBytes)} on the GPU. Close other apps and reopen this one. Nothing was loaded.",
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
            backend = Backend.GPU(),
            visionBackend = Backend.GPU(),
            maxNumImages = 1,
            cacheDir = context.cacheDir.absolutePath,
          )
        ).also { it.initialize() }
      }
      engine = e
      State.Ready(SystemClock.elapsedRealtime() - start).also { state = it }
    } catch (t: Throwable) {
      unavailable("The model could not be loaded on the GPU: ${t.message ?: t.javaClass.simpleName}")
    }
  }

  /**
   * Sends the picture and the instruction in a new conversation and streams the answer text.
   * Thinking is turned off through the chat template's `enable_thinking`: the answer starts
   * right away instead of after a reasoning block.
   */
  fun ask(png: ByteArray, instruction: String): Flow<String> {
    val e = engine ?: error("The model is not loaded.")
    val noThinking = mapOf<String, Any>("enable_thinking" to false)
    val conversation = e.createConversation(
      ConversationConfig(extraContext = noThinking, maxOutputToken = MAX_OUTPUT_TOKENS)
    )
    return conversation
      .sendMessageAsync(Contents.of(Content.ImageBytes(png), Content.Text(instruction)), noThinking)
      .map { message ->
        message.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
      }
      .onCompletion { conversation.close() }
      .flowOn(Dispatchers.IO)
  }

  private fun unavailable(message: String, retry: Boolean = false): State =
    State.Unavailable(message, retry).also { state = it }

  private fun gb(bytes: Long) = String.format(Locale.US, "%.1f GB", bytes / 1e9)
}
