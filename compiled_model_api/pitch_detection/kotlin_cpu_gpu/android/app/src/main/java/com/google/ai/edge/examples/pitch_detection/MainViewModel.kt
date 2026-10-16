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

package com.google.ai.edge.examples.pitch_detection

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [PitchDetector] and exposes a single [UiState] for the tuner screen. On startup it loads
 * the model (from filesDir, pushed via install_to_device.sh) and self-tests on a synthesized 440 Hz
 * sine. While listening, a mic loop runs CREPE on the latest [PitchDetector.WINDOW]-sample window
 * and republishes the note / cents / Hz every frame. The detector reuses native buffers, so model
 * creation and the mic loop share one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val SELF_TEST_HZ = 440.0
    /** A confident pitch is shown only above this activation; below it we report silence. */
    private const val CONFIDENCE_THRESHOLD = 0.5f
    /** Half the note is "in tune" within this many cents of the target. */
    private const val IN_TUNE_CENTS = 5

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var detector: PitchDetector? = null

  /** Flipped off to stop the mic loop; the loop owns start/stop of the [AudioRecord]. */
  private val listening = AtomicBoolean(false)

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      val modelFile = File(context.filesDir, PitchDetector.MODEL)
      if (!modelFile.exists()) {
        val message = context.getString(R.string.error_model_missing, modelFile.absolutePath)
        _uiState.update { it.copy(errorMessage = message) }
        return@launch
      }
      try {
        val loaded = PitchDetector(context)
        detector = loaded
        // Self-test: synthesize a 440 Hz sine (A4) and confirm the pipeline, no mic needed.
        val sine =
          FloatArray(PitchDetector.WINDOW) {
            sin(2.0 * PI * SELF_TEST_HZ * it / PitchDetector.SAMPLE_RATE).toFloat()
          }
        loaded.detect(sine) // warm up GPU
        val pitch = loaded.detect(sine)
        _uiState.update {
          it.copy(
            isModelReady = true,
            statusMessage =
              context.getString(
                R.string.status_ready,
                "${pitch.note}${pitch.octave}",
                pitch.hz,
              ),
          )
        }
      } catch (t: Throwable) {
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_model_load))
        }
      }
    }
  }

  /** Starts or stops the continuous tuner. Caller must hold RECORD_AUDIO before starting. */
  fun toggleListening() {
    if (listening.getAndSet(!listening.get())) {
      // Was listening → the loop sees the cleared flag and tears down the mic.
      _uiState.update { it.copy(isListening = false) }
      return
    }
    _uiState.update { it.copy(isListening = true) }
    viewModelScope.launch(inferenceDispatcher) { listenLoop() }
  }

  /** Reads the mic into a rolling window and republishes the pitch until [listening] clears. */
  private fun listenLoop() {
    val detector = detector ?: return
    val sampleRate = PitchDetector.SAMPLE_RATE
    val window = PitchDetector.WINDOW
    val minBuffer =
      AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
      )
    // UNPROCESSED keeps AGC/NS off for accurate pitch; fall back to MIC if unavailable.
    val record =
      try {
        AudioRecord(
          MediaRecorder.AudioSource.UNPROCESSED,
          sampleRate,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
          maxOf(minBuffer, sampleRate),
        )
      } catch (e: Throwable) {
        AudioRecord(
          MediaRecorder.AudioSource.MIC,
          sampleRate,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
          maxOf(minBuffer, sampleRate),
        )
      }
    val ring = FloatArray(window)
    val chunk = ShortArray(512)
    fun shiftIn(count: Int) {
      if (count <= 0) return
      val k = minOf(count, window)
      System.arraycopy(ring, k, ring, 0, window - k)
      for (i in 0 until k) {
        ring[window - k + i] = chunk[count - k + i] / 32768f
      }
    }
    try {
      record.startRecording()
      while (listening.get()) {
        shiftIn(record.read(chunk, 0, chunk.size)) // blocking: advance one hop
        // Drain any backlog so the window stays current.
        while (true) {
          val m = record.read(chunk, 0, chunk.size, AudioRecord.READ_NON_BLOCKING)
          if (m <= 0) break
          shiftIn(m)
        }
        publish(detector.detect(ring))
      }
    } catch (t: Throwable) {
      _uiState.update {
        it.copy(isListening = false, errorMessage = t.message ?: "Tuner failed")
      }
    } finally {
      runCatching { record.stop() }
      record.release()
    }
  }

  /** Maps one CREPE result to the tuner display: big note, ±50-cent gauge, and Hz line. */
  private fun publish(pitch: PitchDetector.Pitch) {
    if (pitch.confidence < CONFIDENCE_THRESHOLD) {
      _uiState.update {
        it.copy(
          hasPitch = false,
          note = "",
          centsGauge = "",
          hzText = context.getString(R.string.hint_listening),
        )
      }
      return
    }
    val inTune = abs(pitch.cents) <= IN_TUNE_CENTS
    _uiState.update {
      it.copy(
        hasPitch = true,
        isInTune = inTune,
        note = "${pitch.note}${pitch.octave}",
        centsGauge = centsGauge(pitch.cents),
        hzText = context.getString(R.string.hz_reading, pitch.hz, pitch.cents),
      )
    }
  }

  /** A 21-cell gauge over ±50 cents: center bar at 0, a marker at the current offset. */
  private fun centsGauge(cents: Int): String {
    val position = (cents + 50).coerceIn(0, 100) * 20 / 100
    val sb = StringBuilder("♭ ")
    for (i in 0..20) {
      sb.append(if (i == 10) '│' else if (i == position) '●' else '·')
    }
    sb.append(" ♯")
    return sb.toString()
  }

  override fun onCleared() {
    super.onCleared()
    listening.set(false)
    detector?.close()
  }
}
