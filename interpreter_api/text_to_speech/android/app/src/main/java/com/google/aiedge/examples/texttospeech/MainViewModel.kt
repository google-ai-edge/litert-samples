/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.texttospeech

import android.app.Application
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.max

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val helper = KokoroTtsHelper(application)
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var audioTrack: AudioTrack? = null

    fun synthesize() {
        if (_uiState.value.status == Status.RUNNING) return
        _uiState.update { it.copy(status = Status.RUNNING, error = null) }
        viewModelScope.launch(Dispatchers.Default) {
            try {
                if (!helper.ready) helper.setup()
                val result = helper.synthesize()
                _uiState.update {
                    it.copy(
                        status = Status.DONE,
                        inferenceMs = result.inferenceMs,
                        istftMs = result.istftMs,
                        audioSeconds = result.audioSeconds,
                        rtf = result.rtf,
                    )
                }
                play(result.audio)
            } catch (e: Exception) {
                _uiState.update { it.copy(status = Status.ERROR, error = e.message ?: e.toString()) }
            }
        }
    }

    private fun play(data: FloatArray) {
        audioTrack?.release()
        val minBuf = AudioTrack.getMinBufferSize(
            KokoroTtsHelper.SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(KokoroTtsHelper.SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(max(minBuf, data.size * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        audioTrack = track
        track.play()
        track.write(data, 0, data.size, AudioTrack.WRITE_BLOCKING)
    }

    override fun onCleared() {
        audioTrack?.release()
        helper.close()
    }
}
