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

package com.google.ai.edge.examples.speaker_diarization

import android.content.Context
import kotlin.math.max
import kotlin.math.min

/**
 * Simplified pyannote-3.1-style diarization:
 * sliding 10 s windows (5 s hop) -> PyanNet powerset segmentation (ONNX CPU) -> per
 * (window, local-speaker) unit with >=1 s of solo speech -> WeSpeaker embedding on the unit's
 * concatenated solo audio (LiteRT GPU) -> agglomerative clustering (centroid, cosine,
 * threshold 0.7046 from the pyannote/speaker-diarization-3.1 config) -> stitched global timeline.
 */
class Diarizer(ctx: Context) {

    data class Segment(val start: Double, val end: Double, val speaker: Int)
    data class Result(
        val segments: List<Segment>, val numSpeakers: Int, val perSpeaker: Map<Int, Double>)

    companion object {
        const val SR = 16000
        const val WINDOW = SegmentationOnnx.WINDOW      // 160000
        const val HOP = WINDOW / 2                       // 5 s
        const val MIN_SOLO_SEC = 1.0
        const val THRESHOLD = 0.7045654963945799         // cosine distance (3.1 config)
        const val MIN_SEG_SEC = 0.30
        const val MIN_GAP_SEC = 0.15
    }

    private val seg = SegmentationOnnx(ctx)
    private val fbank = Fbank(ctx)
    private val embedder = SpeakerEmbedder(ctx)

    private class SpkUnit(
        val window: Int, val local: Int,
        val soloFrames: MutableList<Int> = mutableListOf(),
        var embedding: FloatArray? = null,
        var cluster: Int = -1,
    )

    fun diarize(pcm: FloatArray, onProgress: (String) -> Unit): Result {
        val nWin = max(1, 1 + (pcm.size - WINDOW + HOP - 1) / HOP)
        val acts = ArrayList<Array<FloatArray>>(nWin)
        for (w in 0 until nWin) {
            onProgress("Segmenting ${w + 1}/$nWin…")
            val x = FloatArray(WINDOW)
            val s = w * HOP
            val n = min(WINDOW, pcm.size - s)
            if (n > 0) System.arraycopy(pcm, s, x, 0, n)
            acts.add(seg.run(x))
        }
        val frames = seg.frames                          // per 10 s window (589)
        val frameSec = WINDOW.toDouble() / SR / frames

        // units with enough solo speech
        val units = ArrayList<SpkUnit>()
        for (w in 0 until nWin) {
            val a = acts[w]
            for (s in 0 until SegmentationOnnx.MAX_LOCAL_SPEAKERS) {
                val u = SpkUnit(w, s)
                for (t in 0 until frames) {
                    var others = 0f
                    for (o in 0 until SegmentationOnnx.MAX_LOCAL_SPEAKERS)
                        if (o != s) others += a[t][o]
                    if (a[t][s] > 0f && others == 0f) u.soloFrames.add(t)
                }
                if (u.soloFrames.size * frameSec >= MIN_SOLO_SEC) units.add(u)
            }
        }
        if (units.isEmpty()) return Result(emptyList(), 0, emptyMap())

        // embeddings from concatenated solo audio (tile-pad to the fixed 5 s window)
        for ((i, u) in units.withIndex()) {
            onProgress("Embedding ${i + 1}/${units.size}…")
            val solo = FloatArray(SpeakerEmbedder.SAMPLES)
            var pos = 0
            outer@ while (pos < solo.size) {              // tile if shorter than 5 s
                for (t in u.soloFrames) {
                    val a = u.window * HOP + (t * frameSec * SR).toInt()
                    val b = min(a + (frameSec * SR).toInt(), pcm.size)
                    for (p in a until b) {
                        if (pos >= solo.size) break@outer
                        solo[pos++] = pcm[p]
                    }
                }
                if (u.soloFrames.isEmpty()) break
            }
            u.embedding = embedder.embed(fbank.compute(solo))
        }

        onProgress("Clustering…")
        cluster(units)
        val numSpeakers = units.mapNotNull { it.cluster.takeIf { c -> c >= 0 } }.distinct().size

        // stitch: each window owns its central region; paint active frames with cluster colors
        val gridSec = frameSec
        val total = pcm.size.toDouble() / SR
        val nGrid = (total / gridSec).toInt() + 1
        val grid = IntArray(nGrid) { -1 }
        val byWindow = units.groupBy { it.window }
        for (w in 0 until nWin) {
            val a = acts[w]
            val own = byWindow[w] ?: continue
            val regionStart = if (w == 0) 0.0 else w * HOP.toDouble() / SR + 2.5
            val regionEnd = if (w == nWin - 1) total else w * HOP.toDouble() / SR + 7.5
            for (u in own) {
                if (u.cluster < 0) continue
                for (t in 0 until frames) {
                    if (a[t][u.local] <= 0f) continue
                    // paint the frame as an interval (a center-point paint leaves periodic
                    // 1-cell holes from float quantization)
                    val t0 = max(w * HOP.toDouble() / SR + t * frameSec, regionStart)
                    val t1 = min(w * HOP.toDouble() / SR + (t + 1) * frameSec, regionEnd)
                    if (t1 <= t0) continue
                    var g = (t0 / gridSec).toInt()
                    while (g * gridSec < t1 && g < nGrid) {
                        if (g >= 0) grid[g] = u.cluster
                        g++
                    }
                }
            }
        }

        // grid -> merged segments
        val segments = ArrayList<Segment>()
        var cur = -1
        var start = 0.0
        for (g in 0..nGrid) {
            val spk = if (g < nGrid) grid[g] else -1
            if (spk != cur) {
                if (cur >= 0) segments.add(Segment(start, g * gridSec, cur))
                cur = spk
                start = g * gridSec
            }
        }
        val merged = postProcess(segments)
        val per = HashMap<Int, Double>()
        for (s in merged) per[s.speaker] = (per[s.speaker] ?: 0.0) + (s.end - s.start)
        return Result(merged, numSpeakers, per)
    }

    /** Agglomerative clustering: centroid linkage, cosine distance, stop at THRESHOLD. */
    private fun cluster(units: List<SpkUnit>) {
        val embs = units.map { it.embedding!! }
        val clusters = units.indices.map { mutableListOf(it) }.toMutableList()

        fun centroid(c: List<Int>): FloatArray {
            val v = FloatArray(SpeakerEmbedder.DIM)
            for (i in c) for (d in v.indices) v[d] += embs[i][d]
            var n = 0f
            for (d in v.indices) n += v[d] * v[d]
            n = kotlin.math.sqrt(n) + 1e-9f
            for (d in v.indices) v[d] /= n
            return v
        }

        while (clusters.size > 1) {
            var bi = -1
            var bj = -1
            var best = Double.MAX_VALUE
            val cents = clusters.map { centroid(it) }
            for (i in clusters.indices) for (j in i + 1 until clusters.size) {
                var cos = 0.0
                for (d in 0 until SpeakerEmbedder.DIM) cos += cents[i][d] * cents[j][d]
                val dist = 1.0 - cos
                if (dist < best) {
                    best = dist
                    bi = i
                    bj = j
                }
            }
            if (best > THRESHOLD) break
            clusters[bi].addAll(clusters[bj])
            clusters.removeAt(bj)
        }
        for ((label, c) in clusters.withIndex()) for (i in c) units[i].cluster = label
    }

    private fun postProcess(segs: List<Segment>): List<Segment> {
        // merge same-speaker segments separated by a tiny gap, drop very short blips
        val out = ArrayList<Segment>()
        for (s in segs.sortedBy { it.start }) {
            val last = out.lastOrNull()
            if (last != null && last.speaker == s.speaker && s.start - last.end <= MIN_GAP_SEC) {
                out[out.size - 1] = Segment(last.start, s.end, s.speaker)
            } else {
                out.add(s)
            }
        }
        return out.filter { it.end - it.start >= MIN_SEG_SEC }
    }

    fun close() {
        seg.close()
        embedder.close()
    }
}
