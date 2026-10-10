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
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/Endpointer.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

/**
 * Cuts a stream of mono PCM into utterances by frame energy, for a transcriber that takes one
 * window per call. Feed consecutive chunks of any size; the state is kept between calls.
 *
 * A frame of [frameMs] is voiced when its RMS is at least [startRms]. Speech starts after [startMs]
 * of consecutive voiced frames ([Event.SpeechStart]); the utterance begins [preRollMs] before the
 * first of those frames (as much of it as was heard since the stream started or the last utterance
 * ended: a soft onset below the start level is kept). It ends once [hangoverMs] of consecutive
 * unvoiced frames have passed, and the [Event.Utterance] carries the audio from the pre-roll to the
 * end of the hangover. An utterance that reaches [maxUtteranceMs] (rounded down to whole frames) is
 * cut there and emitted (never longer); the frames after the cut go through the start rule again.
 * [flush] returns the open utterance at stream end.
 *
 * Not thread-safe: one stream, one caller.
 */
class Endpointer(
  val sampleRate: Int = 16000,
  val startRms: Float = DEFAULT_START_RMS,
  val startMs: Int = 100,
  val hangoverMs: Int = 800,
  val maxUtteranceMs: Int = 16000,
  val frameMs: Int = 20,
  val preRollMs: Int = 300,
) {
  companion object {
    /** A voice spoken toward the phone; quieter sound through a speaker needs a lower level. */
    const val DEFAULT_START_RMS = 0.02f
  }

  sealed class Event {
    object SpeechStart : Event() {
      override fun toString() = "SpeechStart"
    }

    /**
     * The utterance from the pre-roll to the end of the hangover, never longer than
     * maxUtteranceMs.
     */
    data class Utterance(val pcm: FloatArray) : Event() {
      override fun equals(other: Any?) = other is Utterance && pcm.contentEquals(other.pcm)

      override fun hashCode() = pcm.contentHashCode()

      override fun toString() = "Utterance(${pcm.size} samples)"
    }
  }

  private val frameSamples = sampleRate * frameMs / 1000
  private val startFrames = (startMs + frameMs - 1) / frameMs
  private val hangoverFrames = (hangoverMs + frameMs - 1) / frameMs
  private val preRollFrames = if (preRollMs <= 0) 0 else (preRollMs + frameMs - 1) / frameMs

  /**
   * Whole frames only: the frame that reaches the maximum is appended whole, so a cut never drops
   * part of a frame.
   */
  private val maxSamples =
    if (frameSamples > 0) {
      (sampleRate.toLong() * maxUtteranceMs / 1000 / frameSamples * frameSamples).toInt()
    } else {
      0
    }
  private val minMeanSquare = startRms.toDouble() * startRms

  init {
    require(frameSamples > 0 && startFrames > 0 && hangoverFrames > 0 && preRollMs >= 0) {
      "frame, start and hangover lengths must be positive, and the pre-roll not negative"
    }
    require(maxSamples >= (startFrames + preRollFrames) * frameSamples) {
      "maxUtteranceMs must hold the pre-roll and the start run " +
        "(${(startFrames + preRollFrames) * frameMs} ms)"
    }
  }

  private val pending = FloatArray(frameSamples)
  private var pendingN = 0
  private var inSpeech = false

  /**
   * While idle: the last frames heard (at most the pre-roll and the start run), oldest first from
   * [idleHead]. The voiced run is always the newest [runFrames] of them, so on speech start the
   * whole ring is the pre-roll followed by the run.
   */
  private val idle = FloatArray((preRollFrames + startFrames) * frameSamples)
  private val idleCap = preRollFrames + startFrames
  private var idleHead = 0
  private var idleFrames = 0
  private var runFrames = 0
  private var utterance = FloatArray(0)
  private var utteranceN = 0
  private var silentFrames = 0

  /** Call with consecutive chunks of the stream. */
  fun feed(chunk: FloatArray): List<Event> {
    val events = ArrayList<Event>(2)
    var i = 0
    while (i < chunk.size) {
      val n = minOf(frameSamples - pendingN, chunk.size - i)
      System.arraycopy(chunk, i, pending, pendingN, n)
      pendingN += n
      i += n
      if (pendingN == frameSamples) {
        frame(events)
        pendingN = 0
      }
    }
    return events
  }

  /**
   * The open utterance at stream end (with the samples of an unfinished frame), or null when no
   * speech is open. Resets the state.
   */
  fun flush(): Event.Utterance? {
    val open =
      if (inSpeech) {
        append(pending, 0, pendingN)
        Event.Utterance(utterance.copyOf(utteranceN))
      } else {
        null
      }
    reset()
    return open
  }

  fun reset() {
    pendingN = 0
    inSpeech = false
    clearIdle()
    utteranceN = 0
    silentFrames = 0
  }

  private fun clearIdle() {
    idleHead = 0
    idleFrames = 0
    runFrames = 0
  }

  private fun frame(events: MutableList<Event>) {
    var ss = 0.0
    for (k in 0 until frameSamples) {
      ss += pending[k].toDouble() * pending[k]
    }
    val voiced = ss / frameSamples >= minMeanSquare
    if (!inSpeech) {
      // Keep the frame either way: an unvoiced one is pre-roll for a later run.
      val slot = (idleHead + idleFrames) % idleCap
      System.arraycopy(pending, 0, idle, slot * frameSamples, frameSamples)
      if (idleFrames < idleCap) {
        idleFrames++
      } else {
        idleHead = (idleHead + 1) % idleCap
      }
      runFrames = if (voiced) runFrames + 1 else 0
      if (runFrames < startFrames) {
        return
      }
      inSpeech = true
      silentFrames = 0
      utteranceN = 0
      events += Event.SpeechStart
      for (f in 0 until idleFrames) {
        append(idle, ((idleHead + f) % idleCap) * frameSamples, frameSamples)
      }
      clearIdle()
    } else {
      append(pending, 0, frameSamples)
      silentFrames = if (voiced) 0 else silentFrames + 1
    }
    if (utteranceN >= maxSamples || silentFrames >= hangoverFrames) {
      events += Event.Utterance(utterance.copyOf(utteranceN))
      inSpeech = false
      utteranceN = 0
      silentFrames = 0
    }
  }

  /**
   * Appends up to the max length. Whole frames always fit (the max is a whole number of frames,
   * checked after each one); so does the unfinished frame at [flush].
   */
  private fun append(src: FloatArray, from: Int, n: Int) {
    val take = minOf(n, maxSamples - utteranceN)
    if (take <= 0) {
      return
    }
    if (utterance.size < utteranceN + take) {
      val size = minOf(maxSamples, maxOf(utteranceN + take, utterance.size * 2, sampleRate))
      utterance = utterance.copyOf(size)
    }
    System.arraycopy(src, from, utterance, utteranceN, take)
    utteranceN += take
  }
}
