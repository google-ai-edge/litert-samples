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
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decode any audio Uri to mono 44.1 kHz float PCM (MediaExtractor + MediaCodec, synchronous). */
object AudioDecoder {

    fun decode(ctx: Context, uri: Uri, maxSeconds: Int): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(ctx, uri, null)
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if ((f.getString(MediaFormat.KEY_MIME) ?: "").startsWith("audio/")) {
                track = i
                format = f
                break
            }
        }
        check(track >= 0 && format != null) { "No audio track in the selected file." }
        extractor.selectTrack(track)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        var srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val pcm = ByteArrayOutputStream()
        val info = MediaCodec.BufferInfo()
        var sawInEos = false
        var sawOutEos = false
        val maxSrcBytes = maxSeconds.toLong() * srcRate * channels * 2 + (1 shl 20)
        while (!sawOutEos && pcm.size() < maxSrcBytes) {
            if (!sawInEos) {
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val buf = codec.getInputBuffer(inIdx)!!
                    val n = extractor.readSampleData(buf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInEos = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }
            val outIdx = codec.dequeueOutputBuffer(info, 10_000)
            if (outIdx >= 0) {
                val buf = codec.getOutputBuffer(outIdx)!!
                val bytes = ByteArray(info.size)
                buf.position(info.offset)
                buf.get(bytes)
                pcm.write(bytes)
                codec.releaseOutputBuffer(outIdx, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    sawOutEos = true
                }
            } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val f = codec.outputFormat
                srcRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            }
        }
        codec.stop()
        codec.release()
        extractor.release()

        // PCM16 -> mono float
        val bb = ByteBuffer.wrap(pcm.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
        val nFrames = bb.remaining() / 2 / channels
        val mono = FloatArray(nFrames)
        for (i in 0 until nFrames) {
            var acc = 0f
            for (c in 0 until channels) {
                acc += bb.short / 32768f
            }
            mono[i] = acc / channels
        }
        return resampleLinear(mono, srcRate, NoiseSuppressor.SR, maxSeconds)
    }

    private fun resampleLinear(x: FloatArray, from: Int, to: Int, maxSeconds: Int): FloatArray {
        val cap = maxSeconds * to
        if (from == to) return if (x.size <= cap) x else x.copyOf(cap)
        val n = minOf((x.size.toLong() * to / from).toInt(), cap)
        val out = FloatArray(n)
        val r = from.toDouble() / to
        for (i in 0 until n) {
            val p = i * r
            val i0 = p.toInt().coerceAtMost(x.size - 1)
            val i1 = (i0 + 1).coerceAtMost(x.size - 1)
            val fr = (p - i0).toFloat()
            out[i] = x[i0] * (1f - fr) + x[i1] * fr
        }
        return out
    }
}
