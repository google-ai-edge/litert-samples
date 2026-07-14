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

package com.google.ai.edge.examples.speech_enhancement

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the CMGAN [NoiseSuppressor] and exposes a single [UiState]. On startup it loads the model
 * from filesDir (pushed via install_to_device.sh). A clip enters either from the mic (record an
 * unprocessed noisy take) or from a picked audio/video file (decoded by [AudioDecoder]); the CMGAN
 * graph enhances it and the ViewModel plays back the noisy and the enhanced tracks through an
 * [AudioTrack]. The graph reuses native buffers, so the model call, the mic capture that feeds it,
 * and playback all run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TAG = "CMGAN"

    /** Longest clip we record or decode, in seconds. */
    private const val MAX_SECONDS = 30

    /** Mic read granularity, in float samples. */
    private const val RECORD_BUFFER_SAMPLES = 1600

    /** Playback peak-normalization target, so quiet enhanced clips are still audible. */
    private const val PLAYBACK_PEAK = 0.9f

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var noiseSuppressor: NoiseSuppressor? = null
  @Volatile private var player: AudioTrack? = null

  /** Flipped from the UI thread to stop the mic loop that polls it on the worker. */
  @Volatile private var recording = false

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        noiseSuppressor = NoiseSuppressor(context)
        _uiState.update {
          it.copy(isModelReady = true, statusMessage = context.getString(R.string.status_ready))
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load", t)
        _uiState.update {
          it.copy(errorMessage = t.message ?: context.getString(R.string.error_load_failed))
        }
      }
    }
  }

  /**
   * Toggles mic recording. The first tap starts an unprocessed capture on the worker; a second tap
   * flips [recording] off (directly, not through the confined dispatcher, so the running capture is
   * not blocked behind it) and the finished take is enhanced. Caller must hold RECORD_AUDIO.
   */
  fun toggleRecord() {
    if (recording) {
      recording = false
      return
    }
    val state = _uiState.value
    if (!state.isModelReady || state.isEnhancing) return
    recording = true
    _uiState.update {
      it.copy(
        isRecording = true,
        errorMessage = null,
        statusMessage = context.getString(R.string.status_recording, MAX_SECONDS),
      )
    }
    viewModelScope.launch(inferenceDispatcher) {
      val captured = recordClip()
      _uiState.update { it.copy(isRecording = false) }
      if (captured.size >= NoiseSuppressor.SR) {
        enhance(captured)
      } else {
        _uiState.update { it.copy(statusMessage = context.getString(R.string.status_too_short)) }
      }
    }
  }

  /** Decodes a picked audio/video [uri] and enhances it. */
  fun enhanceUri(uri: Uri) {
    val state = _uiState.value
    if (!state.isModelReady || state.isRecording || state.isEnhancing) return
    viewModelScope.launch(inferenceDispatcher) {
      try {
        _uiState.update {
          it.copy(statusMessage = context.getString(R.string.status_decoding), errorMessage = null)
        }
        val clip = AudioDecoder.decode(context, uri, MAX_SECONDS)
        check(clip.size >= NoiseSuppressor.SR) {
          context.getString(R.string.error_clip_too_short)
        }
        enhance(clip)
      } catch (t: Throwable) {
        Log.e(TAG, "pick", t)
        _uiState.update {
          it.copy(
            isEnhancing = false,
            errorMessage = context.getString(R.string.status_failed, t.message ?: ""),
          )
        }
      }
    }
  }

  /** Plays back the noisy source clip. */
  fun playNoisy() = play(_uiState.value.noisy)

  /** Plays back the CMGAN-enhanced clip. */
  fun playEnhanced() = play(_uiState.value.clean)

  /** Runs CMGAN on [clip], reporting per-chunk progress, and keeps both tracks for playback. */
  private fun enhance(clip: FloatArray) {
    val suppressor = noiseSuppressor ?: return
    _uiState.update { it.copy(isEnhancing = true, noisy = clip, errorMessage = null) }
    val startNanos = System.nanoTime()
    val enhanced =
      suppressor.enhance(clip) { chunk, total ->
        _uiState.update {
          it.copy(statusMessage = context.getString(R.string.status_enhancing, chunk, total))
        }
      }
    val elapsedMs = (System.nanoTime() - startNanos) / 1_000_000
    val seconds = clip.size / NoiseSuppressor.SR
    Log.i(TAG, "enhanced ${seconds}s in ${elapsedMs}ms")
    _uiState.update {
      it.copy(
        isEnhancing = false,
        clean = enhanced,
        statusMessage =
          context.getString(R.string.status_enhanced, seconds, elapsedMs / 1000.0),
      )
    }
  }

  /** Peak-normalizes [data] and plays it back through a fresh static [AudioTrack]. */
  private fun play(data: FloatArray?) {
    data ?: return
    viewModelScope.launch(inferenceDispatcher) {
      stopPlayback()
      var peak = 1e-6f
      for (v in data) {
        if (abs(v) > peak) {
          peak = abs(v)
        }
      }
      val scaled = FloatArray(data.size) { data[it] / peak * PLAYBACK_PEAK }
      val track =
        AudioTrack.Builder()
          .setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build()
          )
          .setAudioFormat(
            AudioFormat.Builder()
              .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
              .setSampleRate(NoiseSuppressor.SR)
              .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
              .build()
          )
          .setBufferSizeInBytes(scaled.size * 4)
          .setTransferMode(AudioTrack.MODE_STATIC)
          .build()
      track.write(scaled, 0, scaled.size, AudioTrack.WRITE_BLOCKING)
      track.play()
      player = track
    }
  }

  /** Captures up to [MAX_SECONDS] of mono 16 kHz unprocessed float PCM until [recording] clears. */
  private fun recordClip(): FloatArray {
    val sampleRate = NoiseSuppressor.SR
    val minBuffer =
      AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
      )
    val recorder =
      AudioRecord(
        MediaRecorder.AudioSource.UNPROCESSED,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
        maxOf(minBuffer, sampleRate * 2),
      )
    val out = FloatArray(sampleRate * MAX_SECONDS)
    var total = 0
    try {
      recorder.startRecording()
      val buffer = FloatArray(RECORD_BUFFER_SAMPLES)
      while (recording && total < out.size) {
        val read =
          recorder.read(buffer, 0, minOf(buffer.size, out.size - total), AudioRecord.READ_BLOCKING)
        if (read > 0) {
          System.arraycopy(buffer, 0, out, total, read)
          total += read
        }
      }
    } finally {
      recorder.stop()
      recorder.release()
      recording = false
    }
    return out.copyOf(total)
  }

  private fun stopPlayback() {
    player?.let { track ->
      runCatching { track.stop() }
      runCatching { track.release() }
    }
    player = null
  }

  override fun onCleared() {
    super.onCleared()
    recording = false
    stopPlayback()
    noiseSuppressor?.close()
  }
}
