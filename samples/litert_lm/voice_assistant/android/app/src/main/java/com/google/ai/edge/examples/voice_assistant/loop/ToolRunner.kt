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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/ToolRunner.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.llm.ChatEngine
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.ToolCall
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

/**
 * One request to the chat model with tools: the model may call [tools] (run here, the results
 * sent back on the same conversation) for up to [maxToolTurns] rounds, then answers in text.
 *
 * Each [turn] opens its own conversation and closes it at the end, so the assistant does not
 * remember the previous request. One turn at a time per engine (the engine's own rule).
 */
class ToolRunner(
  private val chat: ChatEngine,
  private val tools: List<VoiceTool>,
  /** The system text for a turn, given the local date and time ("Saturday, 2026-10-03 15:04"). */
  private val systemInstruction: (now: String) -> String = { defaultSystemInstruction(it) },
  private val maxToolTurns: Int = MAX_TOOL_TURNS,
  /** Ask the model to reason in its thought channel. */
  private val thinking: Boolean = false,
) {
  init {
    require(maxToolTurns >= 1) { "maxToolTurns must be at least 1" }
    require(tools.map { it.name }.toSet().size == tools.size) { "tool names must be unique" }
  }

  /**
   * Runs [text] as one request. The flow ends with exactly one [ToolEvent.Done] or
   * [ToolEvent.Failed]; a [VoiceAssistantException] from the model ends it with Failed too.
   * Cancelling the collector cancels the model. [ToolEvent.Text] carries every model turn's text
   * without call markup, in order: a turn that calls tools streams the text before its first call
   * as it comes and the rest of it (after or between the calls) once the turn has ended, before its
   * calls run ([VoiceLoop] speaks all of it).
   */
  fun turn(text: String): Flow<ToolEvent> = flow {
    val t0 = System.nanoTime()
    val now = SimpleDateFormat("EEEE, yyyy-MM-dd HH:mm", Locale.US).format(Date())
    var firstTokenNs = -1L
    var chunks = 0
    var chars = 0
    var calls = 0
    var turns = 0
    var decodeNs = 0L
    fun timing() =
      TurnTiming(
        firstTokenMs = if (firstTokenNs < 0) -1.0 else (firstTokenNs - t0) / 1e6,
        replyMs = (System.nanoTime() - t0) / 1e6,
        chunks = chunks,
        chars = chars,
        toolCalls = calls,
        turns = turns,
        decodeMs = decodeNs / 1e6,
      )
    val session =
      try {
        chat.createConversation(systemInstruction(now), RuntimeToolFormat.providers(tools))
      } catch (e: VoiceAssistantException) {
        emit(ToolEvent.Failed("${e.code}: ${e.message}", timing(), e.code))
        return@flow
      }
    try {
      var message = Message.user(text)
      while (true) {
        turns++
        val buf = StringBuilder()
        var shown = 0
        val runtimeCalls = ArrayList<ToolCall>()
        val sentAt = System.nanoTime()
        var turnLast = -1L
        try {
          session.send(message, thinking).collect { m ->
            val at = System.nanoTime()
            turnLast = at
            if (firstTokenNs < 0) {
              firstTokenNs = at
            }
            chunks++
            for (thought in m.channels.values) {
              if (thought.isNotEmpty()) {
                chars += thought.length
                emit(ToolEvent.Thinking(thought))
              }
            }
            val piece = m.text
            if (piece.isNotEmpty()) {
              chars += piece.length
              buf.append(piece)
              val visible = RuntimeToolFormat.visibleLength(buf.toString())
              if (visible > shown) {
                emit(ToolEvent.Text(buf.substring(shown, visible)))
                shown = visible
              }
            }
            runtimeCalls += m.toolCalls
          }
        } catch (e: VoiceAssistantException) {
          val soFar = if (buf.isEmpty()) "" else " (text so far: ${buf.take(300)})"
          emit(ToolEvent.Failed("${e.code}: ${e.message}$soFar", timing(), e.code))
          return@flow
        }
        if (turnLast >= 0) {
          decodeNs += turnLast - sentAt
        }
        val all = buf.toString()
        val parsed = RuntimeToolFormat.parse(all, runtimeCalls)
        if (parsed.error != null) {
          emit(ToolEvent.Failed("${parsed.error}: ${all.take(300)}", timing()))
          return@flow
        }
        // What the turn said that is not shown yet: an end held back while it might have been
        // markup, and the text after or between the calls.
        val unshown = unshownText(all, shown, parsed.said)
        if (unshown.isNotEmpty()) {
          emit(ToolEvent.Text(unshown))
        }
        if (parsed.calls.isEmpty()) {
          emit(ToolEvent.Done(parsed.said, timing()))
          return@flow
        }
        if (turns > maxToolTurns) {
          val names = parsed.calls.joinToString { it.name }
          emit(
            ToolEvent.Failed(
              "the model still calls tools after $maxToolTurns rounds: $names",
              timing(),
            )
          )
          return@flow
        }
        val responses = ArrayList<Content>(parsed.calls.size)
        for (c in parsed.calls) {
          calls++
          val tc = System.nanoTime()
          val result = execute(c)
          emit(ToolEvent.ToolCalled(c.name, c.args, result, (System.nanoTime() - tc) / 1e6, turns))
          responses += RuntimeToolFormat.response(c.name, result)
        }
        message = Message.tool(Contents.of(responses))
      }
    } finally {
      withContext(NonCancellable) { session.closeAndJoin() }
    }
  }

  private suspend fun execute(c: ParsedCall): String {
    val tool =
      tools.firstOrNull { it.name == c.name }
        ?: return "Error: unknown tool ${c.name} (available: ${tools.joinToString { it.name }})"
    return try {
      tool.call(c.args)
    } catch (e: CancellationException) {
      throw e
    } catch (e: Exception) {
      "Error: ${e.message ?: e.javaClass.simpleName}"
    }
  }

  companion object {
    /** Model rounds that may call tools before the request fails. */
    const val MAX_TOOL_TURNS = 4

    /** The phone assistant's system text, with the local date and time. */
    fun defaultSystemInstruction(now: String): String =
      "You are a phone assistant. The current date and time is $now. " +
        "Use the provided tools to read the calendar, set alarms and timers, and add events. " +
        "Use get_current_datetime when you need the current local date and time."
  }
}

/**
 * The text parts of a [Message] joined in order ("" when it carries none). Each streamed chunk is
 * incremental; content diverted into a channel is not in it (`channels`).
 */
internal val Message.text: String
  get() = contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }

/** What [ToolRunner.turn] streams, in order; it ends with one [Done] or [Failed]. */
sealed interface ToolEvent {
  /** A piece of the model's reasoning (its thought channel), incremental. */
  data class Thinking(val delta: String) : ToolEvent

  /** A piece of the model's visible text, incremental, without call markup (any model turn's). */
  data class Text(val delta: String) : ToolEvent

  /**
   * A call that ran: its arguments as the model gave them, the text sent back, how long the tool
   * took, and the model turn (from 1) that asked for it.
   */
  data class ToolCalled(
    val name: String,
    val args: Map<String, Any?>,
    val result: String,
    val ms: Double,
    val turn: Int = 0,
  ) : ToolEvent

  /** The answer: the last model turn's text without call markup. */
  data class Done(val reply: String, val timing: TurnTiming) : ToolEvent

  /**
   * The request could not finish: a malformed call, too many rounds, or the model's error ([code]
   * set for a [VoiceAssistantException]).
   */
  data class Failed(val reason: String, val timing: TurnTiming, val code: String? = null) :
    ToolEvent
}

/**
 * One request's wall clock on the device, not a benchmark. [firstTokenMs]: from
 * [ToolRunner.turn]'s start (the conversation is created inside it) to the first chunk of the
 * first model turn, -1 when none came; [replyMs]: to the end; [decodeMs]: the sum over model turns
 * of the time from sending the turn's message to its last chunk (prefill included; a turn without
 * chunks adds nothing). [chunks] and [chars] count what was streamed (text and reasoning; call
 * markup the runtime parses itself is never streamed).
 */
data class TurnTiming(
  val firstTokenMs: Double,
  val replyMs: Double,
  val chunks: Int,
  val chars: Int,
  val toolCalls: Int,
  val turns: Int,
  val decodeMs: Double,
)
