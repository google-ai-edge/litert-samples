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

package com.google.ai.edge.examples.text_to_speech_streaming

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Owns [KittenG2P] and [KittenSynthesizer] and exposes a single [UiState].
 *
 * Streaming: [speak] splits the text into sentences ([SentenceChunker]) and runs a
 * producer/consumer pair — the synthesis loop (confined worker, the graphs reuse native buffers)
 * pushes each sentence's PCM into a channel while a playback coroutine drains it into a streaming
 * [AudioTrack]. Sentence N+1 synthesizes while sentence N plays, so time-to-first-audio is one
 * sentence and playback never waits for the full text.
 */
class MainViewModel(private val context: Context) : ViewModel() {

  companion object {
    // The first sentence is deliberately short: streaming starts after one sentence, so it is
    // what the headline time-to-first-audio number measures.
    const val DEFAULT_TEXT =
      "Hi there! I am a tiny voice with fifteen million parameters. " +
        "Everything you hear is generated on this phone, fully offline, " +
        "streaming sentence by sentence while the rest is still being synthesized."
    private const val TAG = "KittenTTS"

    /** 200 ms per AudioTrack write, so a blocked write never delays cancellation for long. */
    private const val WRITE_SLICE_FRAMES = KittenSynthesizer.SAMPLE_RATE / 5

    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  private var g2p: KittenG2P? = null
  private var tts: KittenSynthesizer? = null
  private var speakJob: Job? = null

  @OptIn(ExperimentalCoroutinesApi::class)
  private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

  private val _uiState =
    MutableStateFlow(
      UiState(
        statusMessage = "Loading KittenTTS…",
        voices = KittenSynthesizer.VOICES,
        selectedVoice = KittenSynthesizer.VOICES.indexOf("expr-voice-5-m").coerceAtLeast(0),
      )
    )
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  init {
    viewModelScope.launch(inferenceDispatcher) {
      try {
        val loadStart = SystemClock.elapsedRealtime()
        val gp = KittenG2P(context).also { g2p = it }
        val t = KittenSynthesizer(context).also { tts = it }
        // Warm up XNNPACK packing + JIT once so the first tap measures steady-state TTFA.
        t.synthesize(gp.phonemize("warm up,"), voice = 0, textLength = 8)
        val ms = SystemClock.elapsedRealtime() - loadStart
        _uiState.update {
          it.copy(
            isModelReady = true,
            modelMegabytes = t.modelBytes / 1e6,
            statusMessage =
              "KittenTTS nano ready in $ms ms — fully offline, try airplane mode.\n" +
                "Type text and tap Speak.",
          )
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load failed", t)
        _uiState.update {
          it.copy(errorMessage = "${t.message}\n\nPush models first:\n  ./install_to_device.sh")
        }
      }
    }
  }

  fun selectVoice(index: Int) {
    _uiState.update { it.copy(selectedVoice = index) }
  }

  /** Stops the current utterance (if any) and streams [text] sentence by sentence. */
  fun speak(text: String) {
    val gp = g2p ?: return
    val t = tts ?: return
    speakJob?.cancel()
    val tapTime = SystemClock.elapsedRealtime()
    speakJob =
      viewModelScope.launch(inferenceDispatcher) {
        val voice = _uiState.value.selectedVoice
        val sentences = SentenceChunker.chunk(text)
        if (sentences.isEmpty()) return@launch
        _uiState.update {
          it.copy(
            isSpeaking = true,
            errorMessage = null,
            statusMessage = "Synthesizing…",
            metrics = Metrics(sentencesTotal = sentences.size),
          )
        }

        val pcmChannel = Channel<FloatArray>(Channel.UNLIMITED)
        val track = newAudioTrack()
        val player =
          launch(Dispatchers.IO) {
            var framesWritten = 0L
            try {
              track.play()
              for (pcm in pcmChannel) {
                // Sliced blocking writes so cancellation (Stop / new tap) stays responsive.
                var offset = 0
                while (offset < pcm.size && isActive) {
                  val count = minOf(WRITE_SLICE_FRAMES, pcm.size - offset)
                  track.write(pcm, offset, count, AudioTrack.WRITE_BLOCKING)
                  offset += count
                }
                framesWritten += offset
                if (!isActive) break
              }
              // Drain: wait until the last buffered frame has actually been played.
              while (isActive && track.playbackHeadPosition < framesWritten) {
                delay(50)
              }
            } finally {
              runCatching { track.pause() }
              runCatching { track.flush() }
              track.release()
            }
          }

        try {
          var synthMs = 0L
          var audioSamples = 0L
          var ttfaMs = 0L
          for ((index, sentence) in sentences.withIndex()) {
            // Explicit check: sends to an UNLIMITED channel never suspend, so without this a
            // cancelled job would keep synthesizing to the end before noticing.
            ensureActive()
            val ids = gp.phonemize(sentence)
            if (ids.isEmpty()) continue
            val result = t.synthesize(ids, voice, sentence.length)
            if (index == 0) ttfaMs = SystemClock.elapsedRealtime() - tapTime
            synthMs += result.synthMs
            audioSamples += result.audio.size
            pcmChannel.send(result.audio)
            val audioSeconds = audioSamples.toDouble() / KittenSynthesizer.SAMPLE_RATE
            Log.i(
              TAG,
              "sentence ${index + 1}/${sentences.size}: ${result.tokens} tokens " +
                "${result.frames} frames ${result.synthMs} ms",
            )
            _uiState.update {
              it.copy(
                statusMessage = "Streaming…",
                metrics =
                  Metrics(
                    ttfaMs = ttfaMs,
                    rtf = synthMs / 1000.0 / audioSeconds,
                    audioSeconds = audioSeconds,
                    synthMs = synthMs,
                    sentencesDone = index + 1,
                    sentencesTotal = sentences.size,
                  ),
              )
            }
          }
          pcmChannel.close()
          player.join()
          _uiState.update { it.copy(isSpeaking = false, statusMessage = "Done — tap Speak again.") }
        } catch (e: Throwable) {
          pcmChannel.close()
          player.cancel()
          if (e is CancellationException) throw e
          Log.e(TAG, "synthesis failed", e)
          _uiState.update {
            it.copy(isSpeaking = false, errorMessage = e.message ?: "Synthesis failed")
          }
        }
      }
  }

  fun stop() {
    speakJob?.cancel()
    speakJob = null
    _uiState.update { it.copy(isSpeaking = false, statusMessage = "Stopped.") }
  }

  private fun newAudioTrack(): AudioTrack {
    val minBytes =
      AudioTrack.getMinBufferSize(
        KittenSynthesizer.SAMPLE_RATE,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
      )
    return AudioTrack(
      AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
      AudioFormat.Builder()
        .setSampleRate(KittenSynthesizer.SAMPLE_RATE)
        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build(),
      // A generous stream buffer so per-sentence writes rarely block the synthesis loop.
      maxOf(minBytes, KittenSynthesizer.SAMPLE_RATE * 4),
      AudioTrack.MODE_STREAM,
      AudioManager.AUDIO_SESSION_ID_GENERATE,
    )
  }

  override fun onCleared() {
    super.onCleared()
    speakJob?.cancel()
    tts?.close()
    g2p?.close()
  }
}
