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

package com.google.ai.edge.examples.audio_source_separation

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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [TigerSeparator] and exposes a single [UiState]. On startup it opens the three TIGER-DnR
 * graphs from filesDir (pushed via install_to_device.sh). The user either picks an audio/video clip
 * or records up to 15 s from the mic; the clip is decoded to mono 44.1 kHz PCM and split into
 * Dialogue / Sound effects / Music stems, which are played back individually through an
 * [AudioTrack]. The graphs reuse native buffers, so the mic capture, the separation and the
 * playback all run on one confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TAG = "TIGER"

    /** Seconds of audio kept from a picked clip (3 chunks: 12.06 s window, 10 s hop). */
    const val MAX_SECONDS = 32

    /** Upper bound on a mic recording. */
    private const val RECORD_SECONDS = 15

    /** Mic read chunk (samples), matching the reference capture loop. */
    private const val RECORD_BUFFER = 4410

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var separator: TigerSeparator? = null
  private var player: AudioTrack? = null
  @Volatile private var recording = false

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        separator = TigerSeparator(context) // fails fast if the models are not installed yet
        _uiState.update {
          it.copy(isModelReady = true, statusMessage = context.getString(R.string.status_ready))
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load failed", t)
        _uiState.update {
          it.copy(
            errorMessage =
              context.getString(R.string.error_load, t.message ?: "Failed to load models")
          )
        }
      }
    }
  }

  /** Decode a picked audio/video [uri] to mono 44.1 kHz PCM and separate it. */
  fun separateFromUri(uri: Uri) {
    if (!_uiState.value.isModelReady || _uiState.value.isSeparating || _uiState.value.isRecording) {
      return
    }
    viewModelScope.launch(inferenceDispatcher) {
      _uiState.update {
        it.copy(
          stems = emptyList(),
          errorMessage = null,
          statusMessage = context.getString(R.string.status_decoding),
        )
      }
      val pcm =
        try {
          val decoded = AudioDecoder.decode(context, uri, MAX_SECONDS)
          check(decoded.size >= TigerSeparator.SR) {
            context.getString(R.string.error_clip_too_short)
          }
          decoded
        } catch (t: Throwable) {
          Log.e(TAG, "decode failed", t)
          _uiState.update { it.copy(errorMessage = t.message ?: "Failed to decode clip") }
          return@launch
        }
      runSeparation(pcm)
    }
  }

  /**
   * Toggle the mic recording. The first tap starts capturing up to [RECORD_SECONDS] s; a second tap
   * stops early. The captured clip is separated once it is at least 1 s long. The caller must hold
   * the RECORD_AUDIO permission.
   */
  fun record() {
    if (recording) { // second tap = stop early
      recording = false
      return
    }
    if (!_uiState.value.isModelReady || _uiState.value.isSeparating) return
    recording = true
    _uiState.update {
      it.copy(
        isRecording = true,
        errorMessage = null,
        statusMessage = context.getString(R.string.status_recording),
      )
    }
    viewModelScope.launch(inferenceDispatcher) {
      val pcm = recordClip()
      _uiState.update { it.copy(isRecording = false) }
      if (pcm.size >= TigerSeparator.SR) {
        runSeparation(pcm)
      } else {
        _uiState.update { it.copy(statusMessage = context.getString(R.string.status_too_short)) }
      }
    }
  }

  /** Start (or restart) playback of [track]'s PCM through a static [AudioTrack]. */
  fun play(track: StemTrack) {
    viewModelScope.launch(inferenceDispatcher) {
      stopPlayback()
      val data = track.samples
      val audioTrack =
        AudioTrack.Builder()
          .setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build()
          )
          .setAudioFormat(
            AudioFormat.Builder()
              .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
              .setSampleRate(TigerSeparator.SR)
              .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
              .build()
          )
          .setBufferSizeInBytes(data.size * 4)
          .setTransferMode(AudioTrack.MODE_STATIC)
          .build()
      audioTrack.write(data, 0, data.size, AudioTrack.WRITE_BLOCKING)
      audioTrack.play()
      player = audioTrack
    }
  }

  /** Separate [pcm] into stems on the GPU, publishing progress and the playable result. */
  private fun runSeparation(pcm: FloatArray) {
    val s = separator ?: return
    _uiState.update { it.copy(isSeparating = true, errorMessage = null) }
    try {
      val startNs = System.nanoTime()
      val separated =
        s.separate(pcm) { stem, chunk, total ->
          val progress = context.getString(R.string.status_separating, stem, chunk, total)
          _uiState.update { it.copy(statusMessage = progress) }
        }
      val ms = (System.nanoTime() - startNs) / 1_000_000
      val seconds = pcm.size / TigerSeparator.SR
      Log.i(TAG, "separated ${seconds}s in ${ms}ms")
      _uiState.update {
        it.copy(
          isSeparating = false,
          stems = buildStems(pcm, separated),
          statusMessage =
            context.getString(R.string.status_separated, seconds, (ms / 1000.0).toString()),
        )
      }
    } catch (t: Throwable) {
      Log.e(TAG, "separate failed", t)
      _uiState.update {
        it.copy(isSeparating = false, errorMessage = t.message ?: "Separation failed")
      }
    }
  }

  /** Wrap the mixture and each separated stem as a labelled, playable [StemTrack]. */
  private fun buildStems(mixture: FloatArray, separated: List<FloatArray>): List<StemTrack> {
    val labelRes =
      mapOf(
        "dialog" to R.string.stem_dialog,
        "effect" to R.string.stem_effect,
        "music" to R.string.stem_music,
      )
    val tracks = ArrayList<StemTrack>(1 + separated.size)
    tracks.add(StemTrack(context.getString(R.string.stem_mixture), mixture))
    for (i in TigerSeparator.STEMS.indices) {
      val key = TigerSeparator.STEMS[i]
      tracks.add(StemTrack(context.getString(labelRes.getValue(key)), separated[i]))
    }
    return tracks
  }

  /**
   * Capture up to [RECORD_SECONDS] s of mono 44.1 kHz float PCM from the mic, returning only the
   * samples actually recorded. Stops early when [recording] is cleared.
   */
  private fun recordClip(): FloatArray {
    val sr = TigerSeparator.SR
    val min =
      AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
    val recorder =
      AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sr,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
        maxOf(min, sr * 2),
      )
    val out = FloatArray(sr * RECORD_SECONDS)
    var total = 0
    try {
      recorder.startRecording()
      val buf = FloatArray(RECORD_BUFFER)
      while (recording && total < out.size) {
        val read =
          recorder.read(buf, 0, minOf(buf.size, out.size - total), AudioRecord.READ_BLOCKING)
        if (read > 0) {
          System.arraycopy(buf, 0, out, total, read)
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
    player?.let {
      runCatching {
        it.stop()
        it.release()
      }
    }
    player = null
  }

  override fun onCleared() {
    super.onCleared()
    recording = false
    stopPlayback()
    separator?.close()
  }
}
