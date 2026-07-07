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

package com.google.ai.edge.examples.music_transcription

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Basic Pitch (Spotify, ICASSP 2022) music transcription on the LiteRT CompiledModel GPU —
 * fully GPU incl. the conv-based CQT front-end. 2 s 22 050 Hz windows (with 30-frame overlap,
 * official inference scheme) -> onset/note posteriorgrams [172, 88] -> note events.
 */
class Transcriber(ctx: Context) : Closeable {

    companion object {
        const val SR = 22050
        const val N_SAMPLES = 43844                 // model window
        const val N_FRAMES = 172
        const val N_NOTES = 88
        const val FFT_HOP = 256
        const val OVERLAP = 30 * FFT_HOP            // 7680 samples (official)
        const val HOP = N_SAMPLES - OVERLAP         // 36164
        const val MIDI_OFFSET = 21                  // A0
        const val FRAME_SEC = FFT_HOP.toDouble() / SR
    }

    data class Note(val startSec: Double, val endSec: Double, val midi: Int, val amplitude: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "basicpitch.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** pcm mono 22 050 Hz -> full-track note/onset posteriorgrams [framesTotal][88]. */
    fun posteriorgrams(pcm: FloatArray, onProgress: (Int, Int) -> Unit): Pair<Array<FloatArray>, Array<FloatArray>> {
        val nWin = if (pcm.size <= N_SAMPLES) 1 else 1 + ((pcm.size - N_SAMPLES) + HOP - 1) / HOP
        // official stitching: keep the CENTER frames of each window (drop overlap/2 at joints)
        val framesTotal = (pcm.size.toDouble() / FFT_HOP).toInt() + 1
        val note = Array(framesTotal) { FloatArray(N_NOTES) }
        val onset = Array(framesTotal) { FloatArray(N_NOTES) }
        val x = FloatArray(N_SAMPLES)
        for (w in 0 until nWin) {
            onProgress(w + 1, nWin)
            val start = w * HOP
            java.util.Arrays.fill(x, 0f)
            val n = minOf(N_SAMPLES, pcm.size - start)
            if (n > 0) System.arraycopy(pcm, start, x, 0, n)
            inBuf[0].writeFloat(x)
            model.run(inBuf, outBuf)
            // outputs ordered: contour [172*264], note [172*88], onset [172*88]
            val noteW = outBuf[1].readFloat()
            val onsetW = outBuf[2].readFloat()
            val f0 = start / FFT_HOP
            val keepFrom = if (w == 0) 0 else OVERLAP / FFT_HOP / 2          // 15
            for (t in keepFrom until N_FRAMES) {
                val g = f0 + t
                if (g >= framesTotal) break
                for (k in 0 until N_NOTES) {
                    note[g][k] = noteW[t * N_NOTES + k]
                    onset[g][k] = onsetW[t * N_NOTES + k]
                }
            }
        }
        return Pair(note, onset)
    }

    /** Simple note-event decoding: onset-triggered, sustained while note posterior stays high. */
    fun decode(note: Array<FloatArray>, onset: Array<FloatArray>,
               onsetTh: Float = 0.5f, frameTh: Float = 0.3f, minFrames: Int = 3): List<Note> {
        val events = ArrayList<Note>()
        val active = IntArray(N_NOTES) { -1 }
        val peak = FloatArray(N_NOTES)
        for (t in note.indices) {
            for (k in 0 until N_NOTES) {
                val on = active[k] >= 0
                if (!on && onset[t][k] >= onsetTh && note[t][k] >= frameTh) {
                    active[k] = t
                    peak[k] = note[t][k]
                } else if (on) {
                    if (note[t][k] < frameTh) {
                        if (t - active[k] >= minFrames)
                            events.add(Note(active[k] * FRAME_SEC, t * FRAME_SEC, k + MIDI_OFFSET, peak[k]))
                        active[k] = -1
                    } else if (note[t][k] > peak[k]) peak[k] = note[t][k]
                }
            }
        }
        for (k in 0 until N_NOTES) if (active[k] >= 0 && note.size - active[k] >= minFrames)
            events.add(Note(active[k] * FRAME_SEC, note.size * FRAME_SEC, k + MIDI_OFFSET, peak[k]))
        return events.sortedBy { it.startSec }
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}
