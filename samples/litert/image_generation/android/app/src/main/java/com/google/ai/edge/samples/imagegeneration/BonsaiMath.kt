// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Host-side math for the Bonsai pipeline (port of the iOS app ../ios/Sources/
// BonsaiMath.swift), kept Android-free so it can be cross-checked on the JVM
// against the recorded pipeline fixtures.

package com.google.ai.edge.samples.imagegeneration

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

object BonsaiMath {
    const val SEQ = 256
    const val TOKENS = 1024        // 512x512 -> 32x32 patch grid
    const val LAT_GRID = 32
    const val PACKED_CHANNELS = 128

    /** FLUX.2-klein sigma schedule (generate.py flowmatch_sigmas): linspace
     *  shifted by the empirical mu, exponential time-shift, timestep == sigma.
     *  steps=4 reproduces the device-fixture manifest sigmas exactly. */
    fun sigmas(steps: Int): FloatArray {
        val m200 = 0.00016927 * TOKENS + 0.45666666
        val m10 = 8.73809524e-05 * TOKENS + 1.89833333
        val a = (m200 - m10) / 190.0
        val mu = a * steps + (m200 - 200.0 * a)
        val out = FloatArray(steps + 1)
        for (i in 0 until steps) {
            val lin = 1.0 - i * (1.0 - 1.0 / steps) / maxOf(steps - 1, 1)
            out[i] = (exp(mu) / (exp(mu) + (1.0 / lin - 1.0))).toFloat()
        }
        return out
    }

    /** Image-token position ids: [0, h, w, 0] over the 32x32 grid -> (1024, 4). */
    fun imgIds(): FloatArray {
        val out = FloatArray(TOKENS * 4)
        for (h in 0 until LAT_GRID) for (w in 0 until LAT_GRID) {
            val base = (h * LAT_GRID + w) * 4
            out[base + 1] = h.toFloat()
            out[base + 2] = w.toFloat()
        }
        return out
    }

    /** Text-token position ids: [0, 0, 0, i] -> (256, 4). */
    fun txtIds(): FloatArray {
        val out = FloatArray(SEQ * 4)
        for (i in 0 until SEQ) out[i * 4 + 3] = i.toFloat()
        return out
    }

    /** Seeded standard-normal noise (1, 1024, 128). SplitMix64 + Box-Muller —
     *  the SAME stream as the iOS app (identical algorithm and constants), so
     *  (prompt, seed, steps) reproduces the same image across platforms. */
    fun noise(seed: Long): FloatArray {
        var state = seed
        fun next(): Long {
            state += -0x61c8864680b583ebL          // 0x9E3779B97F4A7C15
            var z = state
            z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L   // 0xBF58476D1CE4E5B9
            z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L   // 0x94D049BB133111EB
            return z xor (z ushr 31)
        }
        // (0, 1], never 0 for ln(); (next() ushr 11) is uniform in [0, 2^53)
        fun uniform(): Double = ((next() ushr 11) + 1.0) / 9007199254740993.0
        val n = TOKENS * PACKED_CHANNELS
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val r = sqrt(-2.0 * ln(uniform()))
            val theta = 2.0 * Math.PI * uniform()
            out[i] = (r * cos(theta)).toFloat()
            if (i + 1 < n) out[i + 1] = (r * sin(theta)).toFloat()
            i += 2
        }
        return out
    }

    /** (1024, 128) packed tokens -> (1, 32, 64, 64) VAE latent: per-PACKED-
     *  channel BN affine first, then the 2x2 patch unfold (packed channel
     *  m = c*4 + i*2 + j lands at z[c, 2h+i, 2w+j] for token (h, w)). */
    fun unpatchify(lat: FloatArray, scale: FloatArray, shift: FloatArray): FloatArray {
        val z = FloatArray(32 * 64 * 64)
        for (h in 0 until LAT_GRID) for (w in 0 until LAT_GRID) {
            val base = (h * LAT_GRID + w) * PACKED_CHANNELS
            for (c in 0 until 32) for (i in 0..1) for (j in 0..1) {
                val m = c * 4 + i * 2 + j
                z[c * 4096 + (2 * h + i) * 64 + (2 * w + j)] =
                    scale[m] * lat[base + m] + shift[m]
            }
        }
        return z
    }
}
