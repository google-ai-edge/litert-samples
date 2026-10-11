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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/VoiceLoop.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import com.google.ai.edge.examples.voice_assistant.asr.Transcriber
import com.google.ai.edge.examples.voice_assistant.llm.ChatEngine
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import kotlinx.coroutines.flow.Flow

/**
 * The voice loop: an utterance in, the phone's answer out of the loudspeaker. [transcriber] hears
 * it, [chat] answers through a [ToolRunner] that runs [tools] on the way, and [speaker] says the
 * answer sentence by sentence while it streams ([SentenceSplitter]'s chunks), each sentence played
 * by [VoiceLoopConfig.player] while the next one is synthesized.
 *
 * A turn's flow: [Event.Heard], [Event.Thinking], then [Event.ToolCalled] and [Event.Speaking] as
 * they happen, [Event.Error] when something failed, and [Event.Done] last. What is said: the
 * model's text of every turn (without call markup and markdown) until a tool that changes
 * something on the phone ([VoiceTool.isAction]) has run; from then on, with
 * [VoiceLoopConfig.speakActionResults], each action's result instead of the model's words
 * (PhoneTools' results are sentences, "Alarm set for 07:30 (Morning Alarm)"; "Sorry, " and the
 * reason for an `Error:` result), so what is said is what the phone did. When the last model turn
 * says nothing and no action's result was said, the results of the last round's actions (a read's
 * result is data, not said); when nothing at all was said, [VoiceLoopConfig.emptyReplyText]; when
 * the request failed (the round limit, a malformed call, the model's error, the transcriber's),
 * [VoiceLoopConfig.failureText] after what was said, with an Error. A transcript that is blank ends
 * the turn without a reply (Heard with empty text, then Done).
 *
 * Turns run one at a time (a second one waits for the first). Each opens its own conversation
 * (ToolRunner), so the assistant does not remember the previous request. Cancelling a turn's
 * collector stops the model's generation (that conversation is closed), the synthesis after the
 * sentence in progress, and the sound. The loop owns neither the three models nor the player:
 * [closeAndJoin] stops the loop and the sound and leaves them open.
 */
class VoiceLoop(
  val transcriber: Transcriber,
  val chat: ChatEngine,
  val speaker: Speaker,
  val tools: List<VoiceTool> = emptyList(),
  val config: VoiceLoopConfig = VoiceLoopConfig(),
) : AutoCloseable {
  private val engine: LoopEngine

  init {
    require(config.endpointer.sampleRate == transcriber.limits.sampleRate) {
      "the endpointer cuts ${config.endpointer.sampleRate} Hz audio; the transcriber takes " +
        "${transcriber.limits.sampleRate} Hz"
    }
    require(config.endpointer.maxUtteranceMs <= transcriber.limits.windowSeconds * 1000) {
      "the endpointer cuts utterances of up to ${config.endpointer.maxUtteranceMs} ms; the " +
        "transcriber takes ${transcriber.limits.windowSeconds} s"
    }
    config.player?.let {
      require(it.sampleRate == speaker.sampleRate) {
        "the player plays ${it.sampleRate} Hz; the speaker writes ${speaker.sampleRate} Hz"
      }
    }
    config.voice?.let {
      require(it in speaker.voices) {
        "voice '$it' is not one of the speaker's (${speaker.voices.joinToString()})"
      }
    }
    val runner =
      ToolRunner(
        chat,
        tools,
        config.systemInstruction ?: { defaultSystemInstruction(it) },
        config.maxToolTurns,
        config.thinking,
      )
    val port =
      config.player?.let { p ->
        PlayerPort({ p.play(it) }, { p.drain() }, { p.stop() }, { p.firstWriteAtNanos })
      }
    val actions = tools.filter { it.isAction }.mapTo(HashSet()) { it.name }
    engine = LoopEngine(transcriber, speaker, runner::turn, config, port, actions)
  }

  /** One turn from a finished utterance: mono samples in [-1, 1] at the transcriber's rate. */
  fun turn(utterance: FloatArray): Flow<Event> = engine.turn(utterance, null)

  /** One turn from typed text: the transcriber is skipped (Heard carries the text, 0 ms). */
  fun turn(text: String): Flow<Event> = engine.turn(null, text)

  /**
   * Hands-free: [audio] (consecutive chunks at the transcriber's rate, e.g. [MicSource.chunks])
   * goes through [VoiceLoopConfig.endpointer]; each utterance it cuts becomes a [turn]. Chunks
   * that arrive while a turn runs are dropped (no barge-in), and the endpointer starts over after
   * the turn. [Event.Listening] comes first and after every turn, when the loop takes audio again.
   * When [audio] ends, an utterance still open is the last turn. One listen at a time per loop. A
   * stopped listen is not over until its collector's job completes: the turn in progress unwinds
   * first (the model's stop, the conversation's close, the sentence in synthesis), and the loop
   * already takes a new listen meanwhile, a second microphone; start the next listen after that
   * job has completed (VoiceViewModel counts a listen by `isCompleted`, not `isActive`).
   */
  fun listen(audio: Flow<FloatArray>): Flow<Event> = engine.listen(audio)

  /**
   * Stops every turn and listen in progress (their flows end) and the sound, and waits for them;
   * the models stay open.
   */
  suspend fun closeAndJoin() = engine.closeAndJoin()

  /** [closeAndJoin] without the wait. */
  override fun close() = engine.close()

  /** What a turn or [listen] streams. */
  sealed class Event {
    /** [listen] takes audio: before the first utterance and after each turn. */
    object Listening : Event() {
      override fun toString() = "Listening"
    }

    /**
     * The transcript as the model gets it ([VoiceLoopConfig.normalizeTranscript]) or the typed
     * text, the utterance's length and the transcription's wall clock.
     */
    data class Heard(val text: String, val audioMs: Double, val transcribeMs: Double) : Event()

    /** The model is answering; it stays so while tools run. */
    object Thinking : Event() {
      override fun toString() = "Thinking"
    }

    /** A tool ran: the arguments as the model gave them, the text sent back, how long it took. */
    data class ToolCalled(
      val name: String,
      val args: Map<String, Any?>,
      val result: String,
      val ms: Double,
    ) : Event()

    /**
     * A sentence is synthesized and goes to the player: the chunk said, the speaker's ms for it,
     * and for the first sentence of the turn [firstAudioMs], from the end of the utterance (the
     * turn's start for text) to the player's first write, or to this synthesis' end without a
     * player; null for the other sentences.
     */
    data class Speaking(val sentence: String, val synthMs: Double, val firstAudioMs: Double?) :
      Event()

    /** The turn is over (the sound included); always the last event of a turn not cancelled. */
    data class Done(val timing: TurnTiming) : Event()

    /**
     * Something failed: [code] for a model's error (a VoiceAssistantException), null otherwise;
     * the turn goes on to Done.
     */
    data class Error(val code: String?, val message: String) : Event()
  }

  /**
   * One turn's wall clock on the device, not a benchmark, in ms from the end of the utterance (the
   * turn's start for text): [transcribeMs] the transcript was ready (0 for text); [firstTokenMs]
   * the model's first chunk came; [firstSentenceMs] the first sentence to say was complete;
   * [firstAudioMs] the player's first write began (with no player: the first sentence was
   * synthesized); [replyMs] the model's answer was complete; [speakMs] the last sentence was
   * synthesized; [totalMs] the turn ended (with a player: the sound was played out). Null when it
   * did not happen. [toolCalls] and [llmTurns] are the runner's; [heard] the text the model got;
   * [reply] the model's text (every model turn, without call markup), said or not; [spoken] what
   * was said: the text handed to the speaker (the model's words, an action's result, the loop's
   * own texts) without markdown.
   */
  data class TurnTiming(
    val transcribeMs: Double,
    val firstTokenMs: Double?,
    val firstSentenceMs: Double?,
    val firstAudioMs: Double?,
    val replyMs: Double,
    val speakMs: Double?,
    val totalMs: Double,
    val toolCalls: Int,
    val llmTurns: Int,
    val heard: String,
    val reply: String,
    val spoken: String,
  )

  companion object {
    /**
     * [ToolRunner.defaultSystemInstruction] and one line for speech: short spoken sentences, no
     * markdown, no lists.
     */
    fun defaultSystemInstruction(now: String): String =
      ToolRunner.defaultSystemInstruction(now) +
        " Answer in one or two short spoken sentences, no markdown, no lists."
  }
}

/**
 * [VoiceLoop]'s settings. [systemInstruction] (null = [VoiceLoop.defaultSystemInstruction]),
 * [maxToolTurns] and [thinking] go to the [ToolRunner]; [voice] (null = the speaker's first) and
 * [speed] to the speaker; [endpointer] cuts [VoiceLoop.listen]'s audio (one listen at a time uses
 * it); [player] plays the sentences (null = synthesize only, no sound: a caller that plays the
 * audio itself). [emptyReplyText] and [failureText] are said when the model says nothing or the
 * request fails. [speakActionResults]: once an action ([VoiceTool.isAction]) has run, say its
 * result instead of the model's words (false: the model's words, whatever the tools did).
 * [normalizeTranscript]: an all-capitals transcript without punctuation (Zipformer's) goes to the
 * model as a sentence, "SET AN ALARM" as "Set an alarm."; a transcript with lower-case letters
 * keeps its case.
 */
data class VoiceLoopConfig(
  val voice: String? = null,
  val speed: Float = 1f,
  val systemInstruction: ((now: String) -> String)? = null,
  val maxToolTurns: Int = ToolRunner.MAX_TOOL_TURNS,
  val thinking: Boolean = false,
  val endpointer: Endpointer = Endpointer(),
  val player: SpeechPlayer? = null,
  val emptyReplyText: String = "Sorry, I did not get that.",
  val failureText: String = "Sorry, I could not finish that.",
  val speakActionResults: Boolean = true,
  val normalizeTranscript: Boolean = true,
)
