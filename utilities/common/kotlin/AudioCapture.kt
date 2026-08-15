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

package __MODULE_PACKAGE__
// Vendored from common/kotlin/AudioCapture.kt — edit the canonical and run tools/sync_common.py --apply.

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.concurrent.thread

/**
 * Microphone capture loop for speech/audio models (16 kHz mono PCM by default).
 *
 * [start] spawns a daemon thread that delivers normalized float chunks in [-1, 1]
 * to the callback; [stop] ends the loop and releases the recorder. The caller is
 * responsible for holding the `RECORD_AUDIO` runtime permission — constructing
 * [AudioRecord] without it throws [SecurityException].
 */
class AudioCapture(
    private val sampleRate: Int = DEFAULT_SAMPLE_RATE,
    private val chunkSize: Int = DEFAULT_CHUNK_SIZE,
) : AutoCloseable {

    companion object {
        /** The sample rate expected by Whisper/Parakeet/wav2vec2-style models. */
        const val DEFAULT_SAMPLE_RATE = 16_000

        /** 100 ms at 16 kHz — small enough for streaming, large enough to amortize reads. */
        const val DEFAULT_CHUNK_SIZE = 1_600

        private const val PCM_SHORT_SCALE = 32_768f
    }

    private var audioRecord: AudioRecord? = null

    @Volatile
    private var recording = false

    /** True while the capture thread is running. */
    val isRecording: Boolean
        get() = recording

    /**
     * Starts capturing; [onChunk] is invoked on the capture thread with a chunk of
     * normalized samples (the array is only valid inside the callback — copy it if
     * accumulating). No-op if already recording.
     *
     * @throws SecurityException when the RECORD_AUDIO permission is missing
     */
    @SuppressLint("MissingPermission")
    fun start(onChunk: (FloatArray) -> Unit) {
        if (recording) {
            return
        }
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBufferSize, chunkSize * 2) * 2,
        )
        audioRecord = record
        record.startRecording()
        recording = true
        thread(name = "AudioCapture", isDaemon = true) {
            val shorts = ShortArray(chunkSize)
            val floats = FloatArray(chunkSize)
            while (recording) {
                val read = record.read(shorts, 0, chunkSize)
                if (read <= 0) {
                    continue
                }
                for (i in 0 until read) {
                    floats[i] = shorts[i] / PCM_SHORT_SCALE
                }
                onChunk(if (read == chunkSize) floats else floats.copyOf(read))
            }
            // Release on the capture thread so a blocked read() never races release().
            record.stop()
            record.release()
        }
    }

    /** Stops the capture loop; the capture thread releases the recorder on exit. */
    fun stop() {
        recording = false
        audioRecord = null
    }

    override fun close() = stop()
}
