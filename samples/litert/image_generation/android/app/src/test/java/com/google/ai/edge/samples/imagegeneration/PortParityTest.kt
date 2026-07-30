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

// JVM parity tests for the Kotlin ports, against the same artifacts that
// validated the Swift ports: the 26-case Python-tokenizer golden set and the
// recorded Mac pipeline fixtures. Skipped (not failed) when the local
// artifacts are absent, so the app still builds from a bare checkout.

package com.google.ai.edge.samples.imagegeneration

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PortParityTest {
    private val home = System.getProperty("user.home")
    private val golden = File("../../testdata/tok_golden.json")
    private val tokDir = File("$home/models/bonsai-image-4b-tflite/hf_upload/tokenizer")
    private val fixDir = File("$home/models/bonsai-image-4b-tflite/device_fixtures")

    private fun floats(name: String): FloatArray {
        val bytes = File(fixDir, name).readBytes()
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun tokenizerMatchesPythonGolden() {
        assumeTrue(golden.exists() && tokDir.exists())
        val g = JSONObject(golden.readText())
        val tok = QwenTokenizer(
            File(tokDir, "vocab.json").inputStream(),
            File(tokDir, "merges.txt").inputStream()
        )
        val cases = g.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val want = c.getJSONArray("body_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val got = tok.encode("user\n" + c.getString("prompt"))
            assertEquals("case $i: ${c.getString("prompt").take(40)}",
                want.toList(), got.toList())
        }

        // full padded encode on case 0
        val c0 = cases.getJSONObject(0)
        val enc = tok.encodePrompt(c0.getString("prompt"))
        val body = c0.getJSONArray("body_ids")
        val suffix = g.getJSONArray("suffix")
        val real = 1 + body.length() + suffix.length()
        assertEquals(256, enc.ids.size)
        assertEquals(g.getInt("im_start"), enc.ids[0])
        assertEquals(g.getInt("pad"), enc.ids[real])
        assertEquals(1, enc.mask[real - 1])
        assertEquals(0, enc.mask[real])
    }

    @Test
    fun sigmasMatchManifest() {
        assumeTrue(fixDir.exists())
        val manifest = JSONObject(File(fixDir, "manifest.json").readText())
        val want = manifest.getJSONArray("sigmas")
        val got = BonsaiMath.sigmas(manifest.getInt("steps"))
        assertEquals(want.length(), got.size)
        for (i in got.indices) {
            assertTrue("sigma[$i]", Math.abs(got[i] - want.getDouble(i)) < 2e-6)
        }
    }

    @Test
    fun positionIdsMatchFixtures() {
        assumeTrue(fixDir.exists())
        assertEquals(floats("img_ids_f32.bin").toList(), BonsaiMath.imgIds().toList())
        assertEquals(floats("txt_ids_f32.bin").toList(), BonsaiMath.txtIds().toList())
    }

    @Test
    fun eulerAndUnpatchifyMatchFixtures() {
        assumeTrue(fixDir.exists())
        val manifest = JSONObject(File(fixDir, "manifest.json").readText())
        val steps = manifest.getInt("steps")
        val sig = manifest.getJSONArray("sigmas")
        val lat = floats("lat0_f32.bin")
        for (k in 0 until steps) {
            val v = floats("dit_out_${k}_f32.bin")
            val ds = (sig.getDouble(k + 1) - sig.getDouble(k)).toFloat()
            for (i in lat.indices) lat[i] += ds * v[i]
        }
        val a = manifest.getJSONArray("affine_a")
        val b = manifest.getJSONArray("affine_b")
        val z = BonsaiMath.unpatchify(
            lat,
            FloatArray(a.length()) { a.getDouble(it).toFloat() },
            FloatArray(b.length()) { b.getDouble(it).toFloat() }
        )
        val zRef = floats("z_vae_f32.bin")
        assertEquals(zRef.size, z.size)
        var maxErr = 0f
        for (i in z.indices) maxErr = maxOf(maxErr, Math.abs(z[i] - zRef[i]))
        assertTrue("unpatchify max err $maxErr", maxErr < 1e-4f)
    }

    @Test
    fun noiseMatchesSwiftStream() {
        // The iOS app's SplitMix64+Box-Muller stream, seed 42: first 8 values
        // and the last, printed by the Mac Swift harness at %.9e —
        // cross-platform image reproducibility depends on this exact stream.
        val swiftFirst8 = floatArrayOf(
            4.147197604e-01f, 6.526812315e-01f, -8.918862343e-01f, 1.326833606e+00f,
            1.729593039e+00f, -1.883416772e+00f, 5.456204414e-01f, -1.656835794e+00f
        )
        val swiftLast = 2.426872700e-01f
        val n = BonsaiMath.noise(42)
        assertEquals(1024 * 128, n.size)
        for (i in swiftFirst8.indices) {
            assertEquals("noise[$i]", swiftFirst8[i], n[i], 1e-7f)
        }
        assertEquals("noise[last]", swiftLast, n[n.size - 1], 1e-7f)
        var mean = 0.0
        var sq = 0.0
        for (x in n) { mean += x; sq += x.toDouble() * x }
        mean /= n.size
        assertTrue("mean $mean", Math.abs(mean) < 0.02)
        assertTrue("var", Math.abs(sq / n.size - mean * mean - 1.0) < 0.03)
        assertEquals(n.toList(), BonsaiMath.noise(42).toList())
        assertTrue(n.toList() != BonsaiMath.noise(43).toList())
    }
}
