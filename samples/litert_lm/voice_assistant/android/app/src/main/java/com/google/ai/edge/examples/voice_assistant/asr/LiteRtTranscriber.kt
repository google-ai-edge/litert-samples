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
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/LiteRtTranscriber.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.asr

import android.util.Log
import com.google.ai.edge.examples.voice_assistant.Runtime
import com.google.ai.edge.examples.voice_assistant.VoiceAssistantException
import com.google.ai.edge.examples.voice_assistant.handOver
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext

/**
 * One zipformer CTC graph on LiteRT `CompiledModel`: host fbank ([ZipformerFbank]) -> the graph
 * ([ZipformerCtc] inputs) -> host greedy CTC. The fbank runs on `Dispatchers.Default`; every
 * native call runs on the shared LiteRT thread ([Runtime]) with the process-wide `Environment`.
 * One `transcribe` at a time.
 */
internal class LiteRtTranscriber
private constructor(
  override val limits: TranscriberLimits,
  private val contract: ZipformerCtc,
  private val pieces: Map<Int, String>,
  private val classes: Int,
  private val model: CompiledModel,
  private val inputs: List<TensorBuffer>,
  private val outputs: List<TensorBuffer>,
  private val fbankSlot: Int,
  private val biasSlots: List<Int>,
  private val logitsSlot: Int,
) : Transcriber {
  private val fbank = ZipformerFbank()
  private val features = FloatArray(contract.frames * ZipformerFbank.NMEL)
  private val busy = AtomicBoolean(false)
  private val closing = AtomicBoolean(false)
  private val closed = CompletableDeferred<Unit>()

  override suspend fun transcribe(pcm: FloatArray): Transcript {
    checkUsable()
    checkLength(pcm.size, limits)
    if (!busy.compareAndSet(false, true)) {
      throw VoiceAssistantException(
        "MODEL_BUSY",
        "a transcription is already running (one at a time per model)",
      )
    }
    try {
      val t0 = System.nanoTime()
      val real =
        withContext(Dispatchers.Default) {
          for (x in pcm) {
            if (!x.isFinite()) {
              throw VoiceAssistantException("INVALID_INPUT", "the audio has a non-finite sample")
            }
          }
          java.util.Arrays.fill(features, ZipformerFbank.LOG_PAD)
          fbank.compute(pcm, features)
          fbank.frames(pcm.size)
        }
      val t1 = System.nanoTime()
      val valid50 = contract.valid50(real)
      val logits = withContext(Runtime.dispatcher) { run(valid50) }
      val t2 = System.nanoTime()
      val text =
        ZipformerCtc.decode(logits, contract.validOut(valid50), classes, contract.blank, pieces)
      val t3 = System.nanoTime()
      val timing =
        TranscriptTiming(
          featureMs = (t1 - t0) / 1e6,
          inferenceMs = (t2 - t1) / 1e6,
          totalMs = (t3 - t0) / 1e6,
        )
      return Transcript(text, timing)
    } finally {
      busy.set(false)
    }
  }

  /**
   * Writes the features and biases, runs the graph and reads the logits back (the readback waits
   * for the GPU). On the LiteRT thread.
   */
  private fun run(valid50: Int): FloatArray =
    try {
      // A close() may have run on this thread while the features were computed.
      if (closed.isCompleted) {
        throw VoiceAssistantException("MODEL_CLOSED", "the transcriber was closed during the call")
      }
      inputs[fbankSlot].writeFloat(features)
      for (r in 0 until 4) {
        inputs[biasSlots[r]].writeFloat(contract.bias(r, valid50))
      }
      model.run(inputs, outputs)
      outputs[logitsSlot].readFloat()
    } catch (e: VoiceAssistantException) {
      throw e
    } catch (t: Throwable) {
      throw VoiceAssistantException(
        "INFERENCE_FAILED",
        "CompiledModel.run failed: ${t.javaClass.simpleName}: ${t.message}",
        t,
      )
    }

  private fun checkUsable() {
    if (closing.get()) {
      throw VoiceAssistantException("MODEL_CLOSED", "the transcriber is closing or closed")
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
    (inputs + outputs).forEach { runCatching { it.close() } }
    runCatching { model.close() }.onFailure { Log.w(TAG, "transcriber graph close: ${it.message}") }
    closed.complete(Unit)
  }

  companion object {
    private const val TAG = "VoiceAssistant"

    /** The published graphs take one 16 s window at 16 kHz (1600 fbank frames). */
    const val WINDOW_SECONDS = 16.0
    const val BLANK_ID = 0
    val LANGUAGES = listOf("en")

    /**
     * Longer than the window, or shorter than one analysis frame (the reflect padding needs it):
     * INVALID_INPUT.
     */
    fun checkLength(samples: Int, limits: TranscriberLimits) {
      val max = (limits.sampleRate * limits.windowSeconds).toInt()
      if (samples > max) {
        val seconds = "%.2f".format(samples.toDouble() / limits.sampleRate)
        throw VoiceAssistantException(
          "INVALID_INPUT",
          "$samples samples ($seconds s) is longer than the ${limits.windowSeconds} s window; " +
            "split the audio (consecutive windows are not handled by this model)",
        )
      }
      if (samples < ZipformerFbank.WIN) {
        throw VoiceAssistantException(
          "INVALID_INPUT",
          "$samples samples is shorter than one ${ZipformerFbank.WIN}-sample analysis frame",
        )
      }
    }

    /**
     * Opens Zipformer CTC on the GPU at the default precision (what the LiteRT model zoo app
     * runs): reads [tokensFile], compiles [modelFile] on the LiteRT thread and checks the graph
     * against the 16 s window's contract (shapes, the blank id) and the tokens. A mismatch or a
     * compile failure is INITIALIZATION_FAILED; there is no fallback to the CPU.
     */
    suspend fun open(modelFile: File, tokensFile: File): LiteRtTranscriber {
      val sampleRate = ZipformerFbank.SR
      val frames = Math.round(WINDOW_SECONDS * sampleRate / ZipformerFbank.HOP).toInt()
      val limits = TranscriberLimits(sampleRate, WINDOW_SECONDS, LANGUAGES)
      val contract = ZipformerCtc(frames, BLANK_ID)
      val pieces =
        withContext(Dispatchers.IO) {
          try {
            ZipformerCtc.readTokens(tokensFile)
          } catch (t: Throwable) {
            throw VoiceAssistantException(
              "INITIALIZATION_FAILED",
              "tokens failed to load: ${t.javaClass.simpleName}: ${t.message}",
              t,
            )
          }
        }
      Log.i(TAG, "tokens ${tokensFile.name}: ${pieces.size} pieces")
      val t0 = System.nanoTime()
      val transcriber =
        try {
          // The compile blocks until the LiteRT thread is done: wait on an IO thread. A caller
          // cancelled meanwhile gets nothing, and the compiled graph is closed instead of dropped.
          handOver(
            Dispatchers.IO,
            { Runtime.call { compile(modelFile, contract, pieces, limits) } },
          ) {
            it.closeAndJoin()
          }
        } catch (e: VoiceAssistantException) {
          throw e
        } catch (e: CancellationException) {
          throw e
        } catch (t: Throwable) {
          throw VoiceAssistantException(
            "INITIALIZATION_FAILED",
            "CompiledModel failed on the GPU: ${t.javaClass.simpleName}: ${t.message}",
            t,
          )
        }
      Log.i(TAG, "zipformer compile_ms=${(System.nanoTime() - t0) / 1_000_000} on gpu")
      return transcriber
    }

    /** On the LiteRT thread. Throws the runtime's exception unchanged. */
    private fun compile(
      file: File,
      contract: ZipformerCtc,
      pieces: Map<Int, String>,
      limits: TranscriberLimits,
    ): LiteRtTranscriber {
      val options = CompiledModel.Options(Accelerator.GPU)
      val model = CompiledModel.create(file.absolutePath, options, Runtime.environment())
      var ins: List<TensorBuffer> = emptyList()
      var outs: List<TensorBuffer> = emptyList()
      try {
        ins = model.createInputBuffers()
        outs = model.createOutputBuffers()
        val inSizes = ins.map { it.readFloat().size }
        val outSizes = outs.map { it.readFloat().size }
        val fbankSlot = inSizes.indexOf(contract.frames * ZipformerFbank.NMEL)
        val biasSlots = contract.biasLengths.map { inSizes.indexOf(it) }
        val logitsSlot = outSizes.indexOfFirst { it > 0 && it % contract.tOut == 0 }
        if (fbankSlot < 0 || biasSlots.any { it < 0 } || logitsSlot < 0) {
          throw VoiceAssistantException(
            "INITIALIZATION_FAILED",
            "the graph does not match the zipformer_ctc contract for a " +
              "${limits.windowSeconds} s window: expected inputs of " +
              "${contract.frames * ZipformerFbank.NMEL} (fbank ${contract.frames}x" +
              "${ZipformerFbank.NMEL}) and ${contract.biasLengths} floats and an output of " +
              "${contract.tOut} x classes; found inputs $inSizes, outputs $outSizes",
          )
        }
        val classes = outSizes[logitsSlot] / contract.tOut
        if (contract.blank >= classes) {
          throw VoiceAssistantException(
            "INITIALIZATION_FAILED",
            "blank id ${contract.blank} is not one of the graph's $classes classes",
          )
        }
        val missing = (0 until classes).firstOrNull { it !in pieces }
        if (missing != null) {
          throw VoiceAssistantException(
            "INITIALIZATION_FAILED",
            "the graph scores $classes classes but tokens has no piece for id $missing",
          )
        }
        return LiteRtTranscriber(
          limits,
          contract,
          pieces,
          classes,
          model,
          ins,
          outs,
          fbankSlot,
          biasSlots,
          logitsSlot,
        )
      } catch (t: Throwable) {
        (ins + outs).forEach { runCatching { it.close() } }
        runCatching { model.close() }
        throw t
      }
    }
  }
}
