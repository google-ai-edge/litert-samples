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

package com.google.ai.edge.examples.model_zoo.audio

import android.util.Log
import com.google.ai.edge.examples.model_zoo.models.basicpitch.Transcriber
import com.google.ai.edge.examples.model_zoo.models.cmgan.NoiseSuppressor
import com.google.ai.edge.examples.model_zoo.models.crepe.PitchDetector
import com.google.ai.edge.examples.model_zoo.models.dac.DacCodec
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.examples.model_zoo.models.panns.AudioTagger
import com.google.ai.edge.examples.model_zoo.models.tiger.TigerSeparator
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.log10

/** Audio containers and routing only. Numerical inference stays in the model wrappers. */
data class AudioStageProgress(val stem: String, val chunk: Int, val totalChunks: Int)

data class AudioTaskRequest(
  val samples: FloatArray,
  val sampleRate: Int,
  val onProgress: (String, Int, Int) -> Unit = { _, _, _ -> },
)

data class NamedWaveform(val name: String, val samples: FloatArray, val sampleRate: Int)

data class BackendSnapshot(
  val actual: String,
  val fallbackReason: String? = null,
  val details: String = "",
)

data class AudioTaskOutput(
  val summary: String,
  val waveforms: List<NamedWaveform> = emptyList(),
  val metrics: Map<String, Any?> = emptyMap(),
  val inferenceMs: Double,
)

interface AudioTaskEngine : AutoCloseable {
  val backend: BackendSnapshot

  fun run(request: AudioTaskRequest): AudioTaskOutput
}

object BatchAudioTasks {
  val ids =
    setOf(
      "audio-classification",
      "pitch-detection",
      "audio-codec",
      "speech-enhancement",
      "audio-source-separation",
      "music-transcription",
    )

  fun create(taskId: String, directory: File, backend: String): AudioTaskEngine =
    when (taskId) {
      "audio-classification" -> PannsTask(directory, backend)
      "pitch-detection" -> CrepeTask(directory, backend)
      "audio-codec" -> DacTask(directory, backend)
      "speech-enhancement" -> CmganTask(directory, backend)
      "audio-source-separation" -> TigerTask(directory, backend)
      "music-transcription" -> BasicPitchTask(directory, backend)
      else -> error("Unknown audio task: $taskId")
    }
}

internal class AudioCompiler(
  private val preferred: String,
  private val details: String = "",
  private val logFailure: (String, String, Exception) -> Unit = { name, reason, failure ->
    Log.w("ModelZooAudio", "$name GPU compilation failed; using CPU: $reason", failure)
  },
) {
  private val failures = linkedMapOf<String, String>()
  private val placements = linkedMapOf<String, String>()

  init {
    require(preferred in setOf("gpu", "cpu", "mixed"))
  }

  val snapshot: BackendSnapshot
    get() =
      BackendSnapshot(
        when {
          placements.values.all { it == "CPU" } && placements.isNotEmpty() -> "CPU"
          placements.values.any { it == "CPU" } -> "GPU + CPU"
          preferred == "cpu" -> "CPU"
          preferred == "mixed" -> "GPU + CPU"
          else -> "GPU"
        },
        failures
          .takeIf { it.isNotEmpty() }
          ?.entries
          ?.joinToString("\n") { "${it.key}: ${it.value}" },
        details +
          placements
            .takeIf { it.size > 1 }
            ?.entries
            ?.joinToString(prefix = "; ") { "${it.key}=${it.value}" }
            .orEmpty(),
      )

  fun <T> create(name: String, factory: (Accelerator) -> T): T {
    if (preferred == "cpu" || failures.containsKey(name)) {
      val result = factory(Accelerator.CPU)
      placements[name] = "CPU"
      return result
    }
    return try {
      val result = factory(Accelerator.GPU)
      placements[name] = "GPU"
      result
    } catch (failure: Exception) {
      val reason = "${failure.javaClass.name}: ${failure.message.orEmpty()}"
      logFailure(name, reason, failure)
      failures[name] = reason
      val result = factory(Accelerator.CPU)
      placements[name] = "CPU"
      result
    }
  }
}

private fun AudioTaskRequest.atRate(rate: Int): FloatArray {
  require(samples.isNotEmpty() && sampleRate > 0 && samples.all { it.isFinite() }) {
    "Choose or record nonempty finite audio first."
  }
  return WavAudio.resample(samples, sampleRate, rate)
}

private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000.0

private fun number(value: Double, digits: Int = 2) =
  String.format(Locale.US, "%${"." + digits}f", value)

private fun waveMetrics(wave: NamedWaveform): Map<String, Any?> =
  mapOf(
    "name" to wave.name,
    "sampleRate" to wave.sampleRate,
    "samples" to wave.samples.size,
    "seconds" to wave.samples.size.toDouble() / wave.sampleRate,
    "peakAmplitude" to (wave.samples.maxOfOrNull { abs(it) } ?: 0f),
    "finiteSamples" to wave.samples.count { it.isFinite() },
  )

private class PannsTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler =
    AudioCompiler(preferred, "CNN14 GPU; original Kotlin log-mel on host CPU by design")
  private val wrapper = compiler.create("CNN14") { AudioTagger(directory, it) }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val output = wrapper.tag(request.atRate(32000), 5)
    val metrics =
      mapOf(
        "top5" to output.tags.map { mapOf("label" to it.label, "score" to it.prob) },
        "melMs" to output.melMs,
        "gpuMs" to output.gpuMs,
        "inputWindowSeconds" to 10,
      )
    return AudioTaskOutput(
      output.tags.joinToString("\n") { "${it.label}: ${number(it.prob * 100.0)}%" },
      metrics = metrics,
      inferenceMs = elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}

private class CrepeTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler = AudioCompiler(preferred)
  private val wrapper =
    compiler.create("CREPE") { PitchDetector(File(directory, "crepe_full_fp16.tflite"), it) }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val audio = request.atRate(PitchDetector.SAMPLE_RATE)
    // Container framing is I/O adaptation; every frame uses the wrapper's normalization and decoder.
    val hop = 1600
    val curve =
      (audio.indices step hop).map { offset ->
        val frame = FloatArray(PitchDetector.WINDOW)
        val count = minOf(frame.size, audio.size - offset)
        audio.copyInto(frame, 0, offset, offset + count)
        val pitch = wrapper.detect(frame)
        mapOf<String, Any?>(
          "timeSeconds" to offset.toDouble() / 16000,
          "hz" to pitch.hz,
          "confidence" to pitch.confidence,
          "note" to "${pitch.note}${pitch.octave}",
          "cents" to pitch.cents,
        )
      }
    return AudioTaskOutput(
      curve.joinToString("\n") {
        "${number(it["timeSeconds"] as Double)} s: ${number((it["hz"] as Float).toDouble())} Hz · ${it["note"]} · confidence ${number((it["confidence"] as Float).toDouble())}"
      },
      metrics =
        mapOf(
          "curve" to curve,
          "f0Hz" to curve.map { it["hz"] },
          "confidence" to curve.map { it["confidence"] },
          "frameCount" to curve.size,
          "hopSeconds" to 0.1,
        ),
      inferenceMs = elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}

private class DacTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler =
    AudioCompiler(preferred, "GPU encoder/decoder; original host RVQ by design")
  private val wrapper = compiler.create("DAC encoder + decoder") { DacCodec(directory, it) }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val audio = request.atRate(16000)
    val chunks = mutableListOf<FloatArray>()
    var signal = 0.0
    var error = 0.0
    var codeCount = 0
    for (offset in audio.indices step DacCodec.SAMPLES) {
      val input = FloatArray(DacCodec.SAMPLES)
      val count = minOf(input.size, audio.size - offset)
      audio.copyInto(input, 0, offset, offset + count)
      val result = wrapper.roundTrip(input)
      val output = result.audio.copyOf(minOf(count, result.audio.size))
      chunks += output
      codeCount += result.codes.size
      for (i in output.indices) {
        signal += input[i].toDouble() * input[i]
        val difference = input[i].toDouble() - output[i]
        error += difference * difference
      }
    }
    val decoded = FloatArray(chunks.sumOf { it.size })
    var offset = 0
    for (chunk in chunks) {
      chunk.copyInto(decoded, offset)
      offset += chunk.size
    }
    val snr = if (signal > 0.0 && error > 0.0) 10.0 * log10(signal / error) else null
    val waves =
      listOf(NamedWaveform("Original", audio, 16000), NamedWaveform("Decoded", decoded, 16000))
    return AudioTaskOutput(
      audioCodecSummary(decoded.size, 16000, codeCount),
      waves,
      mapOf(
        "waveforms" to waves.map(::waveMetrics),
        "snrDb" to snr,
        "snrAlignment" to
          "Per 1-second chunk, first min(input,output) samples; no lag compensation",
        "codeCount" to codeCount,
      ),
      elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}

private class CmganTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler =
    AudioCompiler(preferred, "GPU spectrogram graph; original host inverse STFT and overlap-add")
  private val wrapper = compiler.create("CMGAN") { NoiseSuppressor(directory, it) }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val original = request.atRate(16000)
    val enhanced = wrapper.enhance(original) { _, _ -> }
    val waves =
      listOf(NamedWaveform("Before", original, 16000), NamedWaveform("Enhanced", enhanced, 16000))
    return AudioTaskOutput(
      "Enhanced ${number(enhanced.size / 16000.0)} s · peak ${number(enhanced.maxOf { abs(it) }.toDouble(), 4)}",
      waves,
      mapOf("waveforms" to waves.map(::waveMetrics)),
      elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}

private class TigerTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler =
    AudioCompiler(
      preferred,
      "The TIGER wrapper loads one stem graph at a time inside each call; timing includes graph compilation and prior stems' inverse STFT",
    )
  private val wrapper =
    TigerSeparator(directory) { file ->
      compiler.create(file.name) {
        CompiledModel.create(file.absolutePath, CompiledModel.Options(it), null)
      }
    }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val audio = request.atRate(44100)
    val stems = wrapper.separate(audio, request.onProgress)
    val waves = stems.mapIndexed { i, pcm -> NamedWaveform(TigerSeparator.STEMS[i], pcm, 44100) }
    return AudioTaskOutput(
      waves.joinToString("\n") { "${it.name}: ${number(it.samples.size / 44100.0)} s" },
      waves,
      mapOf("waveforms" to waves.map(::waveMetrics), "compileInsideRun" to true),
      elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}

private class BasicPitchTask(directory: File, preferred: String) : AudioTaskEngine {
  private val compiler = AudioCompiler(preferred)
  private val wrapper = compiler.create("Basic Pitch") { Transcriber(directory, it) }
  override val backend
    get() = compiler.snapshot

  override fun run(request: AudioTaskRequest): AudioTaskOutput {
    val started = System.nanoTime()
    val (note, onset) = wrapper.posteriorgrams(request.atRate(Transcriber.SR)) { _, _ -> }
    val notes = Transcriber.decode(note, onset)
    val first =
      notes.take(10).map {
        mapOf(
          "pitch" to it.midi,
          "onsetSeconds" to it.startSec,
          "endSeconds" to it.endSec,
          "amplitude" to it.amplitude,
        )
      }
    val summary =
      com.google.ai.edge.examples.model_zoo.ResultCounts.notes(notes.size) +
        "\n" +
        notes.take(10).joinToString("\n") {
          "MIDI ${it.midi}: ${number(it.startSec)}–${number(it.endSec)} s"
        }
    return AudioTaskOutput(
      summary,
      metrics =
        mapOf("noteCount" to notes.size, "first10Notes" to first, "posteriorFrames" to note.size),
      inferenceMs = elapsed(started),
    )
  }

  override fun close() = wrapper.close()
}
