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

package com.google.aiedge.examples.audio_classification

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.isActive

class AudioManager(
    private val sampleRate: Int,
    private val bufferSize: Int,
    private val overlap: Float
) {
    private var audioRecord: AudioRecord? = null

    companion object {
        private const val TAG = "AudioManager"
    }

    @SuppressLint("MissingPermission")
    suspend fun record(): Flow<ShortArray> {
        return flow {
            val buffer = (bufferSize * (1 - overlap)).toInt()
            Log.i(TAG, "bufferSize = $buffer")
            audioRecord = AudioRecord(
                // including MIC, UNPROCESSED, and CAMCORDER.
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                buffer
            )
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                return@flow
            }
            Log.i(TAG, "Successfully initialized AudioRecord")
            val audioBuffer = ShortArray(buffer)
            audioRecord?.startRecording()

            while (currentCoroutineContext().isActive) {
                when (audioRecord?.read(
                    audioBuffer,
                    0,
                    audioBuffer.size
                )) {
                    AudioRecord.ERROR_INVALID_OPERATION -> {
                        Log.w(TAG, "AudioRecord.ERROR_INVALID_OPERATION")
                    }

                    AudioRecord.ERROR_BAD_VALUE -> {
                        Log.w(TAG, "AudioRecord.ERROR_BAD_VALUE")
                    }

                    AudioRecord.ERROR_DEAD_OBJECT -> {
                        Log.w(TAG, "AudioRecord.ERROR_DEAD_OBJECT")
                    }

                    AudioRecord.ERROR -> {
                        Log.w(TAG, "AudioRecord.ERROR")
                    }

                    buffer -> {
                        Log.i(TAG, "record: $audioBuffer")
                        emit(audioBuffer)
                    }
                }
            }
        }
    }

    fun stopRecord() {
        audioRecord?.stop()
    }
}
