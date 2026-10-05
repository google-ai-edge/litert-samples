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
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/LiteRtSpeaker.kt
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/Speak.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.tts

import android.util.Log
import com.google.ai.edge.examples.voice_assistant.Runtime
import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.handOver
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext

/**
 * KittenTTS on classic LiteRT: [KittenG2P] (dictionary + the out-of-dictionary graph) -> the style
 * row for the text's length -> [KittenSynthesizer], with `speed` times the voice's prior. Every
 * native call runs on the shared LiteRT thread ([Runtime]); the G2P runs there whole, so an
 * out-of-dictionary word does not hop threads. One `synthesize` at a time.
 */
internal class LiteRtSpeaker(
  override val voices: List<String>,
  override val sampleRate: Int,
  override val maxChars: Int,
  private val styles: Map<String, NpzVoices.Table>,
  /** The voice's factor on `speed` before the graph (none: 1). */
  private val priors: Map<String, Double>,
  private val g2p: KittenG2P,
  private val neural: KittenNeuralG2P,
  private val synth: KittenSynthesizer,
) : Speaker {
  private val busy = AtomicBoolean(false)
  private val closing = AtomicBoolean(false)
  private val closed = CompletableDeferred<Unit>()

  override suspend fun synthesize(text: String, voice: String?, speed: Float): SpeechAudio {
    checkUsable()
    val name = voice ?: voices[0]
    val table =
      styles[name]
        ?: throw VoiceAssistantException(
          "INVALID_INPUT",
          "unknown voice '$name' (this model has ${voices.joinToString()})",
        )
    if (!speed.isFinite() || speed <= 0f) {
      throw VoiceAssistantException(
        "INVALID_INPUT",
        "speed $speed: expected a finite value above 0",
      )
    }
    val chars = checkText(text, maxChars)
    if (!busy.compareAndSet(false, true)) {
      throw VoiceAssistantException(
        "MODEL_BUSY",
        "a synthesis is already running (one at a time per model)",
      )
    }
    try {
      val t0 = System.nanoTime()
      val ids = withContext(Runtime.dispatcher) { symbols(text) }
      val t1 = System.nanoTime()
      // The pip package's lookup: one style row per text length, the last row for anything longer.
      val style = table.row(minOf(chars, table.rows - 1))
      val out =
        withContext(Runtime.dispatcher) {
          checkNotClosed()
          try {
            synth.synthesize(ids, style, graphSpeed(speed, priors[name] ?: 1.0))
          } catch (t: Throwable) {
            throw VoiceAssistantException(
              "INFERENCE_FAILED",
              "synthesis of ${ids.size} symbols failed: ${t.javaClass.simpleName}: ${t.message}",
              t,
            )
          }
        }
      val t2 = System.nanoTime()
      val timing =
        SpeechTiming(
          g2pMs = (t1 - t0) / 1e6,
          synthMs = (t2 - t1) / 1e6,
          totalMs = (t2 - t0) / 1e6,
          frames = out.frames,
        )
      return SpeechAudio(out.samples, sampleRate, timing)
    } finally {
      busy.set(false)
    }
  }

  override fun phonemeIds(text: String): IntArray {
    checkUsable()
    checkText(text, maxChars)
    return Runtime.call { symbols(text) }
  }

  /**
   * The G2P on the LiteRT thread; text without a symbol to sound (punctuation alone, or nothing the
   * model knows) is INVALID_INPUT.
   */
  private fun symbols(text: String): IntArray {
    checkNotClosed()
    val ids =
      try {
        g2p.ids(text)
      } catch (t: Throwable) {
        throw VoiceAssistantException(
          "INFERENCE_FAILED",
          "g2p failed: ${t.javaClass.simpleName}: ${t.message}",
          t,
        )
      }
    if (!g2p.sounds(ids)) {
      throw VoiceAssistantException(
        "INVALID_INPUT",
        "the text has nothing to say: no symbol the model knows other than punctuation and spaces",
      )
    }
    return ids
  }

  private fun checkUsable() {
    if (closing.get()) {
      throw VoiceAssistantException("MODEL_CLOSED", "the speaker is closing or closed")
    }
  }

  // A close() may have run on the LiteRT thread while this call waited for it.
  private fun checkNotClosed() {
    if (closed.isCompleted) {
      throw VoiceAssistantException("MODEL_CLOSED", "the speaker was closed during the call")
    }
  }

  override fun close() {
    if (closing.compareAndSet(false, true)) {
      Runtime.executor.execute { doClose() }
    }
  }

  override suspend fun closeAndJoin() {
    val first = closing.compareAndSet(false, true)
    withContext(NonCancellable) {
      if (first) {
        Runtime.call { doClose() }
      }
      closed.await()
    }
  }

  private fun doClose() {
    if (closed.isCompleted) {
      return
    }
    runCatching { synth.close() }.onFailure { Log.w(TAG, "speaker graphs close: ${it.message}") }
    runCatching { neural.close() }.onFailure { Log.w(TAG, "g2p graph close: ${it.message}") }
    closed.complete(Unit)
  }

  companion object {
    private const val TAG = "VoiceAssistant"

    /** The kitten vocoder writes 24 kHz. */
    const val SAMPLE_RATE = 24000

    /** The voices of `voices.npz`, the first the default. */
    val VOICES =
      listOf(
        "expr-voice-2-m",
        "expr-voice-2-f",
        "expr-voice-3-m",
        "expr-voice-3-f",
        "expr-voice-4-m",
        "expr-voice-4-f",
        "expr-voice-5-m",
        "expr-voice-5-f",
      )

    /**
     * The publisher's `say.py` factors: `speed` times the voice's factor goes to the graph, so
     * `speed = 1` is say.py's default pace.
     */
    val SPEED_PRIORS =
      mapOf(
        "expr-voice-2-f" to 0.8,
        "expr-voice-2-m" to 0.8,
        "expr-voice-3-m" to 0.8,
        "expr-voice-3-f" to 0.8,
        "expr-voice-4-m" to 0.9,
        "expr-voice-4-f" to 0.8,
        "expr-voice-5-m" to 0.8,
        "expr-voice-5-f" to 0.8,
      )
    const val STYLE_ROWS = 400
    const val CPU_THREADS = 4

    /** The pip package's trim of each chunk's end. */
    const val TAIL_TRIM = 5000
    const val MIN_SAMPLES = 1200

    /** One chunk's limit. */
    const val MAX_CHARS = 400

    /**
     * say.py's `speed * SPEED_PRIORS.get(voice, 1.0)` in double, then float32 for the graph: 1.25
     * x 0.8 is 1.0 exactly.
     */
    fun graphSpeed(speed: Float, prior: Double): Float = (speed.toDouble() * prior).toFloat()

    /**
     * Empty, or longer than [maxChars] code points: INVALID_INPUT. Returns the length in code
     * points (Python's `len`).
     */
    fun checkText(text: String, maxChars: Int): Int {
      if (text.isBlank()) {
        throw VoiceAssistantException("INVALID_INPUT", "the text is empty")
      }
      val chars = text.codePointCount(0, text.length)
      if (chars > maxChars) {
        throw VoiceAssistantException(
          "INVALID_INPUT",
          "$chars characters is longer than one chunk ($maxChars); split the text " +
            "(SentenceSplitter)",
        )
      }
      return chars
    }

    /**
     * Opens KittenTTS nano on the CPU: the style tables, the symbol table ([symbolsJson] reads
     * the app's `symbols.json` asset), the G2P dictionary and meta, then on the LiteRT thread the
     * G2P graph (CompiledModel, CPU, 4 threads) and the three synthesis graphs (Interpreter, 4
     * threads, XNNPACK except on [KittenSynthesizer.XNNPACK_OFF]). [file] gives each file of the
     * catalog entry by its name. A file that does not load is INITIALIZATION_FAILED.
     */
    suspend fun open(file: (String) -> File, symbolsJson: () -> String): LiteRtSpeaker {
      fun <T> stage(stage: String, block: () -> T): T =
        try {
          block()
        } catch (e: VoiceAssistantException) {
          throw e
        } catch (t: Throwable) {
          throw VoiceAssistantException(
            "INITIALIZATION_FAILED",
            "$stage failed to load: ${t.javaClass.simpleName}: ${t.message}",
            t,
          )
        }
      val ms = LinkedHashMap<String, Long>()
      fun <T> timed(key: String, block: () -> T): T {
        val t0 = System.nanoTime()
        return block().also { ms[key] = (System.nanoTime() - t0) / 1_000_000 }
      }

      val loaded =
        withContext(Dispatchers.IO) {
          val npz = timed("voices") { stage("voices") { NpzVoices.read(file("voices.npz")) } }
          val styles =
            VOICES.associateWith { v ->
              val t =
                npz[v]
                  ?: throw VoiceAssistantException(
                    "INITIALIZATION_FAILED",
                    "voices.npz has no voice '$v' (it has ${npz.names.joinToString()})",
                  )
              if (t.rows != STYLE_ROWS || t.dim != KittenSynthesizer.STYLE_DIM) {
                throw VoiceAssistantException(
                  "INITIALIZATION_FAILED",
                  "voices.npz '$v' is ${t.rows} x ${t.dim}, expected $STYLE_ROWS x " +
                    "${KittenSynthesizer.STYLE_DIM}",
                )
              }
              t
            }
          val symbolToId = stage("lexicon") { KittenG2P.readSymbols(symbolsJson()) }
          val meta =
            stage("lexicon") { KittenNeuralG2P.Meta.parse(file("g2p_meta.json").readText()) }
          val dictionary =
            timed("g2p_dict") {
              stage("lexicon") { KittenG2P.readDictionary(file("g2p_dict.txt.gz")) }
            }
          Log.i(
            TAG,
            "kitten lexicon: ${symbolToId.size} symbols, ${dictionary.size} dictionary words, " +
              "${npz.names.size} voices",
          )
          Lexicon(styles, symbolToId, meta, dictionary)
        }

      // The graphs open on the LiteRT thread; wait for them on an IO thread. A caller cancelled
      // meanwhile gets nothing, and the speaker made of them is closed instead of dropped.
      val speaker =
        handOver(
          Dispatchers.IO,
          {
            val (neural, synth) =
              Runtime.call {
                val options =
                  CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
                  }
                val neural =
                  timed("g2p_graph") {
                    stage("g2p_graph") {
                      KittenNeuralG2P.open(
                        file("dp_g2p_matcha_fp16.tflite"),
                        loaded.meta,
                        options,
                        Runtime.environment(),
                      )
                    }
                  }
                val synth =
                  try {
                    stage("interpreter") {
                      KittenSynthesizer.open(
                        file("kitten_predictor.tflite"),
                        file("kitten_prosody.tflite"),
                        file("kitten_vocoder.tflite"),
                        CPU_THREADS,
                        TAIL_TRIM,
                        MIN_SAMPLES,
                        ms,
                        xnnpackOff = KittenSynthesizer.XNNPACK_OFF,
                      )
                    }
                  } catch (t: Throwable) {
                    runCatching { neural.close() }
                    throw t
                  }
                neural to synth
              }
            LiteRtSpeaker(
              VOICES,
              SAMPLE_RATE,
              MAX_CHARS,
              loaded.styles,
              SPEED_PRIORS,
              KittenG2P(loaded.dictionary, loaded.symbolToId, neural::word),
              neural,
              synth,
            )
          },
        ) {
          it.closeAndJoin()
        }
      Log.i(
        TAG,
        "kitten load_ms " + ms.entries.joinToString(" ") { "${it.key}=${it.value}" } +
          " (xnnpack off on ${KittenSynthesizer.XNNPACK_OFF.joinToString("+")}, " +
          "$CPU_THREADS threads)",
      )
      return speaker
    }
  }

  private class Lexicon(
    val styles: Map<String, NpzVoices.Table>,
    val symbolToId: Map<Char, Int>,
    val meta: KittenNeuralG2P.Meta,
    val dictionary: Map<String, String>,
  )
}
