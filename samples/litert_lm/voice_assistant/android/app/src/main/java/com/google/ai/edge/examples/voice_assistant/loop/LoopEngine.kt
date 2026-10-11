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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/LoopEngine.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.asr.Transcriber
import com.google.ai.edge.examples.voice_assistant.loop.VoiceLoop.Event
import com.google.ai.edge.examples.voice_assistant.tts.Speaker
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

/** What the loop needs from a player: [SpeechPlayer]'s calls. */
internal class PlayerPort(
  val play: suspend (FloatArray) -> Unit,
  val drain: suspend () -> Unit,
  val stop: () -> Unit,
  val firstWriteAtNanos: () -> Long,
)

/**
 * [VoiceLoop]'s turns and listen without the chat engine's wiring: [reply] is a ToolRunner's turn.
 * [actions]: the names of the tools whose results are said instead of the model's words
 * ([VoiceTool.isAction]).
 */
internal class LoopEngine(
  private val transcriber: Transcriber?,
  private val speaker: Speaker,
  private val reply: (String) -> Flow<ToolEvent>,
  private val config: VoiceLoopConfig,
  private val player: PlayerPort?,
  private val actions: Set<String> = emptySet(),
) {
  private val turnLock = Mutex()
  private val listening = AtomicBoolean(false)
  private val active: MutableSet<Job> = ConcurrentHashMap.newKeySet()

  @Volatile private var closed = false

  // The turn's clock starts when it has the loop: the wait for an earlier turn is not its time,
  // and that turn's first write to the player is not its first sound.
  fun turn(pcm: FloatArray?, text: String?): Flow<Event> = tracked {
    turnLock.withLock { runTurn(System.nanoTime(), pcm, text) }
  }

  fun listen(audio: Flow<FloatArray>): Flow<Event> = tracked {
    check(listening.compareAndSet(false, true)) {
      "this VoiceLoop is already listening (one listen at a time)"
    }
    try {
      val endpointer = config.endpointer.also { it.reset() }
      val busy = AtomicBoolean(false)
      val utterances = Channel<FloatArray>(Channel.RENDEZVOUS)
      val turns = launch {
        for (pcm in utterances) {
          turn(pcm, null).collect { send(it) }
          busy.set(false)
          send(Event.Listening)
        }
      }
      send(Event.Listening)
      audio.collect { chunk ->
        // A turn is running: the chunk is dropped (no barge-in).
        if (busy.get()) {
          return@collect
        }
        val cut =
          endpointer.feed(chunk).firstOrNull { it is Endpointer.Event.Utterance }
            as Endpointer.Event.Utterance?
        if (cut != null) {
          busy.set(true)
          // What came after the cut in this chunk and everything heard during the turn is not
          // this utterance.
          endpointer.reset()
          utterances.send(cut.pcm)
        }
      }
      if (!busy.get()) {
        endpointer.flush()?.let {
          busy.set(true)
          utterances.send(it.pcm)
        }
      }
      utterances.close()
      turns.join()
    } finally {
      listening.set(false)
    }
  }

  suspend fun closeAndJoin() {
    closed = true
    val jobs = active.toList()
    jobs.forEach { it.cancel() }
    jobs.forEach { it.join() }
    player?.stop?.invoke()
  }

  fun close() {
    closed = true
    active.forEach { it.cancel() }
    player?.stop?.invoke()
  }

  /** A flow whose producer [closeAndJoin] can stop. */
  private fun tracked(block: suspend ProducerScope<Event>.() -> Unit): Flow<Event> = channelFlow {
    check(!closed) { "this VoiceLoop is closed" }
    val job = coroutineContext[Job]!!
    active += job
    try {
      block()
    } finally {
      active -= job
    }
  }

  private suspend fun ProducerScope<Event>.runTurn(t0: Long, pcm: FloatArray?, typed: String?) {
    fun since(at: Long) = (at - t0) / 1e6
    val speech = Speech(this, t0)
    // The model's text (TurnTiming.reply) and the text handed to the speaker (TurnTiming.spoken).
    val replyText = StringBuilder()
    val spokenText = StringBuilder()
    // A text of the loop's own (an action's result, the empty-reply or the failure text), said
    // whole.
    fun say(text: String) {
      if (text.isBlank()) {
        return
      }
      spokenText.join(text, newText = true)
      speech.say(SentenceSplitter.split(text, speaker.maxChars))
    }
    // An action has run: its result was said, and the model's words from here on are not.
    var actionSaid = false
    var heard = ""
    var transcribeMs = 0.0
    var llmAt = 0L
    var replyAt = 0L
    var runnerTiming: TurnTiming? = null
    val calls = ArrayList<ToolEvent.ToolCalled>()
    try {
      if (pcm != null) {
        val asr = checkNotNull(transcriber) { "no transcriber" }
        val transcript =
          try {
            asr.transcribe(pcm)
          } catch (e: VoiceAssistantException) {
            send(Event.Error(e.code, "transcriber: ${e.message}"))
            null
          }
        transcribeMs = since(System.nanoTime())
        if (transcript == null) {
          say(config.failureText)
        } else {
          val text = transcript.text.trim()
          heard = if (config.normalizeTranscript) normalizeTranscript(text) else text
          send(Event.Heard(heard, pcm.size * 1000.0 / asr.limits.sampleRate, transcribeMs))
        }
      } else {
        heard = typed.orEmpty().trim()
        send(Event.Heard(heard, 0.0, 0.0))
      }
      if (heard.isNotEmpty()) {
        send(Event.Thinking)
        val stream = SentenceStream(speaker.maxChars)
        var afterCall = false
        llmAt = System.nanoTime()
        reply(heard).collect { e ->
          when (e) {
            is ToolEvent.Thinking -> {}
            is ToolEvent.Text -> {
              if (e.delta.isNotEmpty()) {
                // A model turn after a tool call starts a new sentence in the record too.
                replyText.join(e.delta, newText = afterCall)
                if (!actionSaid) {
                  spokenText.join(e.delta, newText = afterCall)
                  speech.say(stream.add(e.delta))
                }
                afterCall = false
              }
            }
            is ToolEvent.ToolCalled -> {
              // The end of a model turn ends its sentence.
              speech.say(stream.flush())
              afterCall = true
              calls += e
              send(Event.ToolCalled(e.name, e.args, e.result, e.ms))
              if (config.speakActionResults && e.name in actions) {
                // What the phone did, not what the model says it did.
                actionSaid = true
                say(actionReceipt(e.result))
              }
            }
            is ToolEvent.Done -> {
              replyAt = System.nanoTime()
              runnerTiming = e.timing
              speech.say(stream.flush())
              if (e.reply.isBlank() && !actionSaid) {
                // The model said nothing after its calls: the last round's actions say what was
                // done (a read's result is data, such as get_calendar_events' JSON, not a
                // sentence to say).
                val last = calls.lastOrNull()?.turn
                for (c in calls) {
                  if (c.turn == last && c.name in actions) {
                    say(actionReceipt(c.result))
                  }
                }
              }
              if (spokenText.isBlank()) {
                say(config.emptyReplyText)
              }
            }
            is ToolEvent.Failed -> {
              replyAt = System.nanoTime()
              runnerTiming = e.timing
              speech.say(stream.flush())
              send(Event.Error(e.code, e.reason))
              say(config.failureText)
            }
          }
        }
      }
      speech.finish()
    } catch (e: CancellationException) {
      player?.stop?.invoke()
      throw e
    }
    val end = System.nanoTime()
    val firstAudioAt =
      if (player != null) {
        player.firstWriteAtNanos().takeIf { it >= t0 }
      } else {
        speech.firstSynthAt.takeIf { it > 0 }
      }
    val timing = runnerTiming
    val turnTiming =
      VoiceLoop.TurnTiming(
        transcribeMs = transcribeMs,
        firstTokenMs = timing?.firstTokenMs?.takeIf { it >= 0 }?.let { since(llmAt) + it },
        firstSentenceMs = speech.firstSentenceAt.takeIf { it > 0 }?.let(::since),
        firstAudioMs = firstAudioAt?.let(::since),
        replyMs = if (replyAt > 0) since(replyAt) else transcribeMs,
        speakMs = speech.lastSynthAt.takeIf { it > 0 }?.let(::since),
        totalMs = since(end),
        toolCalls = timing?.toolCalls ?: calls.size,
        llmTurns = timing?.turns ?: 0,
        heard = heard,
        reply = replyText.toString().trim(),
        spoken = stripMarkdown(spokenText.toString()).trim(),
      )
    send(Event.Done(turnTiming))
  }

  /**
   * Appends [piece]; a [newText] (a result, the loop's own text, a model turn after a call) is set
   * off by a space.
   */
  private fun StringBuilder.join(piece: String, newText: Boolean) {
    if (newText && isNotEmpty() && !last().isWhitespace() && !piece.first().isWhitespace()) {
      append(' ')
    }
    append(piece)
  }

  /**
   * One turn's speech: the sentences synthesized in order on one coroutine and, with a player,
   * played on another, so the next sentence is synthesized while one plays. A sentence the speaker
   * finds nothing to say in (INVALID_INPUT: punctuation alone) is skipped; any other speaker or
   * player failure is an Error and ends the speech of the turn (the model and its tools go on).
   */
  private inner class Speech(private val scope: ProducerScope<Event>, private val t0: Long) {
    private val sentences = Channel<String>(Channel.UNLIMITED)
    private val audio = Channel<FloatArray>(Channel.UNLIMITED)

    @Volatile
    var firstSentenceAt = 0L
      private set

    @Volatile
    var firstSynthAt = 0L
      private set

    @Volatile
    var lastSynthAt = 0L
      private set

    @Volatile private var speakerFailed = false

    @Volatile private var playerFailed = false

    private val synth =
      scope.launch {
        for (s in sentences) {
          if (speakerFailed) {
            continue
          }
          val a =
            try {
              speaker.synthesize(s, config.voice, config.speed)
            } catch (e: VoiceAssistantException) {
              if (e.code != "INVALID_INPUT") {
                speakerFailed = true
                scope.send(Event.Error(e.code, "speaker: ${e.message}"))
              }
              continue
            }
          val done = System.nanoTime()
          lastSynthAt = done
          val first = firstSynthAt == 0L
          if (first) {
            firstSynthAt = done
          }
          if (player != null && !playerFailed) {
            audio.send(a.samples)
          }
          val firstAudioMs =
            if (!first) {
              null
            } else if (player == null) {
              (done - t0) / 1e6
            } else {
              awaitFirstWrite()?.let { (it - t0) / 1e6 }
            }
          scope.send(Event.Speaking(s, a.timing.totalMs, firstAudioMs))
        }
        audio.close()
      }

    private val play =
      player?.let { p ->
        scope.launch {
          for (pcm in audio) {
            if (playerFailed) {
              continue
            }
            try {
              p.play(pcm)
            } catch (e: CancellationException) {
              throw e
            } catch (e: Exception) {
              playerFailed = true
              scope.send(Event.Error(null, "player: ${e.javaClass.simpleName}: ${e.message}"))
            }
          }
        }
      }

    /**
     * The player's first write of this turn: the player is idle when the first sentence comes, so
     * it starts within ms.
     */
    private suspend fun awaitFirstWrite(): Long? =
      withTimeoutOrNull(FIRST_WRITE_WAIT_MS) {
        var at = player!!.firstWriteAtNanos()
        while (at < t0 && !playerFailed) {
          delay(1)
          at = player.firstWriteAtNanos()
        }
        if (at >= t0) at else null
      }

    /** [chunks] without markdown ([stripMarkdown]); a chunk left blank is dropped. */
    fun say(chunks: List<String>) {
      for (c in chunks) {
        val text = stripMarkdown(c).trim()
        if (text.isEmpty()) {
          continue
        }
        if (firstSentenceAt == 0L) {
          firstSentenceAt = System.nanoTime()
        }
        sentences.trySend(text)
      }
    }

    /** Every sentence synthesized and, with a player, played out. */
    suspend fun finish() {
      sentences.close()
      synth.join()
      play?.join()
      if (player != null && !playerFailed) {
        player.drain()
      }
    }
  }

  private companion object {
    const val FIRST_WRITE_WAIT_MS = 2_000L
  }
}

/**
 * An action's result as it is said: as it is, or for an `Error:` result "Sorry, " and the reason.
 */
internal fun actionReceipt(result: String): String =
  if (result.startsWith("Error:")) "Sorry, " + result.removePrefix("Error:").trim() else result

private val PRONOUN_I = Regex("\\bi\\b")

/**
 * A transcript as the model gets it with [VoiceLoopConfig.normalizeTranscript]: an all-capitals
 * transcript (Zipformer writes capitals without punctuation) in lower case with the first letter
 * and the word I in capitals; then a full stop when it does not end in a sentence mark. A
 * transcript with a lower-case letter keeps its case.
 */
internal fun normalizeTranscript(text: String): String {
  var t = text.trim()
  if (t.isEmpty()) {
    return t
  }
  if (t.none { it.isLowerCase() }) {
    t = t.lowercase().replace(PRONOUN_I, "I").replaceFirstChar { it.uppercaseChar() }
  }
  if (t.last() !in SentenceSplitter.MARKS) {
    t += "."
  }
  return t
}
