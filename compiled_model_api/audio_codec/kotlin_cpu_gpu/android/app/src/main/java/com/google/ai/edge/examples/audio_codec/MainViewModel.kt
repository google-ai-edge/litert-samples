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

package com.google.ai.edge.examples.audio_codec

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [MimiCodec] and exposes a single [UiState]. On startup it decodes a bundled 2 s clip,
 * loads the four Mimi graphs (from filesDir, pushed via install_to_device.sh), warms up the GPU
 * shaders, and round-trips the clip (encode → quantize → decode) so the screen can A/B the original
 * against the reconstruction. Both waveforms are played back through an [AudioTrack] as a side
 * effect. The graphs reuse native buffers, so every model call — and every playback — runs on one
 * confined worker.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    private const val TAG = "Mimi"
    private const val TEST_AUDIO_ASSET = "test_audio.bin"
    private const val SAMPLE_RATE = 24000 // Mimi operates at 24 kHz.
    private const val CLIP_MILLIS = 2000.0 // Fixed 2 s clip; the RTF denominator.

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var codec: MimiCodec? = null
  private var audioTrack: AudioTrack? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(UiState(statusMessage = context.getString(R.string.status_loading)))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val original = readFloats(context.assets.open(TEST_AUDIO_ASSET).readBytes())
        val loaded = MimiCodec(context)
        codec = loaded
        loaded.roundTrip(original) // warm up the GPU shaders + JIT the RVQ
        val result = loaded.roundTrip(original) // measured pass
        val rtf = (result.encodeMs + result.decodeMs) / CLIP_MILLIS
        Log.i(
          TAG,
          "encode=${result.encodeMs}ms decode=${result.decodeMs}ms " +
            "codes=${result.codes.size} rtf=$rtf",
        )
        _uiState.update {
          it.copy(
            isModelReady = true,
            original = original,
            reconstructed = result.audio,
            statusMessage =
              context.getString(
                R.string.status_success,
                result.encodeMs,
                result.decodeMs,
                rtf,
                MimiRvq.NQ,
                MimiCodec.TC,
                result.codes.size,
              ),
          )
        }
        play(result.audio) // auto-play the reconstruction once, like the original demo
      } catch (t: Throwable) {
        Log.e(TAG, "codec failed", t)
        _uiState.update {
          it.copy(errorMessage = context.getString(R.string.error_model, t.message ?: ""))
        }
      }
    }
  }

  /** Plays the bundled original clip through an [AudioTrack] on the confined worker. */
  fun playOriginal() {
    val audio = _uiState.value.original
    if (audio.isEmpty()) return
    viewModelScope.launch(inferenceDispatcher) { play(audio) }
  }

  /** Plays the codec reconstruction through an [AudioTrack] on the confined worker. */
  fun playReconstructed() {
    val audio = _uiState.value.reconstructed
    if (audio.isEmpty()) return
    viewModelScope.launch(inferenceDispatcher) { play(audio) }
  }

  /** Decodes little-endian float32 PCM samples from the bundled asset bytes. */
  private fun readFloats(bytes: ByteArray): FloatArray {
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(bytes.size / 4) { buffer.float }
  }

  /** Plays a mono float32 waveform at 24 kHz and blocks until it finishes. */
  private fun play(audio: FloatArray) {
    try {
      val track =
        AudioTrack(
          AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
          AudioFormat.Builder()
            .setSampleRate(SAMPLE_RATE)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build(),
          audio.size * 4,
          AudioTrack.MODE_STATIC,
          AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
      audioTrack = track
      track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
      track.play()
      Thread.sleep((audio.size / 24L) + 250)
      track.release()
    } catch (e: Throwable) {
      Log.e(TAG, "play failed: ${e.message}")
    }
  }

  override fun onCleared() {
    super.onCleared()
    audioTrack?.let {
      runCatching { it.stop() }
      runCatching { it.release() }
    }
    codec?.close()
  }
}
