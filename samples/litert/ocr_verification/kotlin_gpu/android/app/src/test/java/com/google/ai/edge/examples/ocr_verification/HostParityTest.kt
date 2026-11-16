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

package com.google.ai.edge.examples.ocr_verification

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The graph inputs the app computes against the Python host's (`ocrverify.fixtures`: one folder per
 * case with the arrays the host wrote, the two decoded pictures and the text's path in `meta.json`;
 * the model folder `ocrverify.models` gives the tokenizer and the tables). Skipped without them.
 * The pictures' decoding is checked on the phone (the app's check mode), not here.
 */
class HostParityTest {
  private val models = System.getProperty("ocrverify.models")?.let(::File)
  private val fixtures = System.getProperty("ocrverify.fixtures")?.let(::File)

  @Test
  fun graphInputsMatchThePythonHost() {
    assumeTrue("set -Pocrverify.models and -Pocrverify.fixtures", models != null && fixtures != null)
    val tables = Bf16Tables(File(models, ModelFiles.TABLES.name))
    val builder = RequestBuilder(QwenBpeTokenizer.fromFile(File(models, ModelFiles.TOKENIZER.name)), tables)
    val report = JSONObject()
    val cases = fixtures!!.listFiles()!!.filter { File(it, "meta.json").isFile }.sortedBy { it.name }
    assertTrue("no case in $fixtures", cases.isNotEmpty())
    var worst = 0.0
    for (dir in cases) {
      val meta = JSONObject(File(dir, "meta.json").readText())
      val text = File(meta.getString("text_file")).readText(Charsets.UTF_8)
      val a = builder.picture(inputRgb(dir, meta, "region"))
      val b = builder.picture(inputRgb(dir, meta, "render"))
      val req = builder.request(a, b, text)
      val row = JSONObject()
      fun exact(name: String, got: ByteArray) {
        val want = File(dir, "$name.bin").readBytes()
        val same = want.contentEquals(got)
        row.put(name, JSONObject().put("exact", same).put("bytes", got.size))
        assertTrue("${dir.name} $name differs", same)
      }
      fun close(name: String, got: FloatArray) {
        val want = floats(File(dir, "$name.bin"))
        assertEquals("${dir.name} $name size", want.size, got.size)
        var maxAbs = 0.0
        var differ = 0
        for (i in got.indices) {
          if (got[i].toRawBits() != want[i].toRawBits()) differ++
          maxAbs = maxOf(maxAbs, Math.abs(got[i].toDouble() - want[i]))
        }
        worst = maxOf(worst, maxAbs)
        row.put(name, JSONObject().put("max_abs", maxAbs).put("values", got.size).put("bit_differ", differ))
        assertTrue("${dir.name} $name max |d| $maxAbs", maxAbs <= 1e-6)
      }
      exact("region_u8", a.resized.bytes)
      exact("render_u8", b.resized.bytes)
      for ((side, pic) in listOf("region" to a, "render" to b)) {
        close("${side}_patches", pic.patches)
        close("${side}_pos", pic.pos)
        close("${side}_cos", pic.cos)
        close("${side}_sin", pic.sin)
      }
      exact("ids", ints(req.ids))
      val zeroA = FloatArray(a.tokens * HostMath.HIDDEN)
      val zeroB = FloatArray(b.tokens * HostMath.HIDDEN)
      close("embeds", builder.embeds(req, zeroA, zeroB, req.tokens))
      val length = meta.getInt("L")
      val feed = builder.decideFeed(req, builder.embeds(req, zeroA, zeroB, length), length)
      close("mrope_cos", feed.getValue("cos"))
      close("mrope_sin", feed.getValue("sin"))
      close("valid", feed.getValue("valid"))
      row.put("T", req.tokens).put("n", "${a.n},${b.n}")
      report.put(dir.name, row)
    }
    report.put("worst_max_abs", worst)
    System.getProperty("ocrverify.report")?.let { File(it).writeText(report.toString(1)) }
    println("PARITY " + report.toString())
    tables.close()
  }

  companion object {
    /** The 8-bit RGB the Python host decoded (Pillow, convert("RGB")), as written next to the arrays. */
    fun inputRgb(dir: File, meta: JSONObject, side: String): HostMath.Rgb {
      val hw = meta.getJSONObject(side).getJSONArray("input_hw")
      return HostMath.Rgb(File(dir, "${side}_input_u8.bin").readBytes(), hw.getInt(1), hw.getInt(0))
    }

    fun floats(file: File): FloatArray {
      val bytes = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      return FloatArray(bytes.remaining()).also { bytes.get(it) }
    }

    fun ints(values: IntArray): ByteArray {
      val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
      values.forEach { buf.putInt(it) }
      return buf.array()
    }
  }
}
