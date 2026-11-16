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

import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File
import java.security.MessageDigest

/** The graph inputs of one request, computed on the phone from the two pictures and the text. */
class Request(
  val region: HostMath.Picture,
  val render: HostMath.Picture,
  val ids: IntArray,
) {
  val tokens: Int
    get() = ids.size
}

/** A request the model is not asked: the host's refusals, with the reason shown to the user. */
class Refused(val status: String, message: String) : Exception(message)

/**
 * Builds [Request]s (no LiteRT: the unit tests run this part on the JVM) and the graphs' input
 * rows, as the Python host of the model repository does.
 */
class RequestBuilder(val tokenizer: QwenBpeTokenizer, val tables: Bf16Tables) {
  init {
    checkTemplate()
  }

  fun picture(rgb: HostMath.Rgb): HostMath.Picture {
    if (rgb.width <= 0 || rgb.height <= 0) throw Refused("unreadable_picture", "a side of 0")
    return HostMath.picture(rgb, tables.posTable)
  }

  fun request(region: HostMath.Picture, render: HostMath.Picture, text: String): Request {
    if ("<|image_pad|>" in text || "<|video_pad|>" in text) {
      throw Refused("literal_pad_token", "the text holds a literal <|image_pad|> or <|video_pad|>")
    }
    val ids = tokenizer.encode(HostMath.buildPrompt(text, region.tokens, render.tokens))
    check(ids.last() == HostMath.POOLED) { "pooled id ${ids.last()}, want ${HostMath.POOLED}" }
    return Request(region, render, ids)
  }

  /** embeds [length, 2048]: table rows, then each picture's tower rows at its image pads. */
  fun embeds(req: Request, rowsRegion: FloatArray, rowsRender: FloatArray, length: Int): FloatArray {
    val out = FloatArray(length * HostMath.HIDDEN)
    var image = 0
    val imageRows = req.region.tokens + req.render.tokens
    for ((t, id) in req.ids.withIndex()) {
      if (id == HostMath.IMAGE_PAD) {
        val (src, row) =
          if (image < req.region.tokens) rowsRegion to image else rowsRender to image - req.region.tokens
        System.arraycopy(src, row * HostMath.HIDDEN, out, t * HostMath.HIDDEN, HostMath.HIDDEN)
        image++
      } else {
        tables.embedRow(id, out, t * HostMath.HIDDEN)
      }
    }
    check(image == imageRows) { "$image image pads for $imageRows picture rows" }
    return out
  }

  fun planes(req: Request): Array<LongArray> =
    HostMath.mropePlanes(
      req.ids,
      listOf(intArrayOf(req.region.gh, req.region.gw), intArrayOf(req.render.gh, req.render.gw)),
    )

  /** The tower's inputs for [pic] in a file of [size] patches: zeros, ones on cos, past n. */
  fun towerFeed(pic: HostMath.Picture, size: Int): Map<String, FloatArray> {
    val n = pic.n
    val patches = FloatArray(size * HostMath.PATCH_DIM)
    System.arraycopy(pic.patches, 0, patches, 0, pic.patches.size)
    val valid = FloatArray(size) { if (it < n) 1f else 0f }
    val pos = FloatArray(size * HostMath.POS_DIM)
    System.arraycopy(pic.pos, 0, pos, 0, pic.pos.size)
    val cos = FloatArray(size * HostMath.ROT_DIM) { 1f }
    System.arraycopy(pic.cos, 0, cos, 0, pic.cos.size)
    val sin = FloatArray(size * HostMath.ROT_DIM)
    System.arraycopy(pic.sin, 0, sin, 0, pic.sin.size)
    return mapOf("patches" to patches, "valid" to valid, "pos" to pos, "cos" to cos, "sin" to sin)
  }

  /** The decision graph's inputs in a file of [length] tokens (the planes continue on the pads). */
  fun decideFeed(req: Request, embeds: FloatArray, length: Int): Map<String, FloatArray> {
    val (cos, sin) = HostMath.mropeTables(HostMath.paddedPlanes(planes(req), length))
    val valid = FloatArray(length) { if (it < req.tokens) 1f else 0f }
    return mapOf("embeds" to embeds, "valid" to valid, "cos" to cos, "sin" to sin)
  }

  companion object {
    /** md5 of the source's template with its placeholders not expanded (score.py's gate). */
    const val TEMPLATE_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"

    fun checkTemplate() {
      val placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
      val s =
        "<|im_start|>user\n$placeholder\nOCR文本：\n{text}\n$placeholder\n" +
          HostMath.INSTRUCTION + HostMath.ASSISTANT_SUFFIX
      val md5 =
        MessageDigest.getInstance("MD5").digest(s.toByteArray(Charsets.UTF_8)).joinToString("") {
          "%02x".format(it)
        }
      check(md5 == TEMPLATE_MD5) { "prompt template md5 $md5, want $TEMPLATE_MD5" }
    }
  }
}

/** One scored request. Times are in ms; [buildMs] is the GPU build of files this request needed. */
class Score(
  val prob1: Double,
  val logits: FloatArray,
  val tokens: Int,
  val patches: IntArray,
  val towerSizes: IntArray,
  val decideSize: Int,
  val ms: LinkedHashMap<String, Double>,
  val buildMs: Double,
) {
  /** The phone's time for this request from the two pictures and the text to the score. */
  val totalMs: Double
    get() = ms.values.sum()
}

/**
 * The scoring host on the phone: the two pictures and the text in, `prob_1` out, every step on
 * the phone (the picture path, the tokenizer and the tables in Kotlin, the graphs on the GPU).
 */
class Verifier(val modelDir: File, keepTowers: Int = 2) : Closeable {
  val tables = Bf16Tables(File(modelDir, ModelFiles.TABLES.name))
  val builder: RequestBuilder
  private val env: Environment
  val towers: BucketGraphs
  val decide: BucketGraphs

  init {
    try {
      builder = RequestBuilder(QwenBpeTokenizer.fromFile(File(modelDir, ModelFiles.TOKENIZER.name)), tables)
      env = Environment.create()
      towers = BucketGraphs(env, modelDir, BucketGraphs.Kind.TOWER, keepTowers)
      decide = BucketGraphs(env, modelDir, BucketGraphs.Kind.DECIDE, 1)
    } catch (e: Throwable) {
      tables.close()
      throw e
    }
  }

  /**
   * Builds the files a typical request needs before the first one comes: the smallest tower and
   * decision files (the ones the example page's lines use). Returns the build times.
   */
  fun warm(towerSizes: List<Int>, decideSize: Int): Map<String, Double> {
    val out = LinkedHashMap<String, Double>()
    for (n in towerSizes) {
      val size = towers.sizeFor(n) ?: continue
      if (towers.isBuilt(size)) continue
      towers.run(size, builder.towerFeed(zeroPicture(), size))
      out["tower_N$size"] = towers.lastBuildMs
    }
    decide.sizeFor(decideSize)?.let { size ->
      if (!decide.isBuilt(size)) {
        val feed =
          mapOf(
            "embeds" to FloatArray(size * HostMath.HIDDEN),
            "valid" to FloatArray(size),
            "cos" to FloatArray(size * HostMath.ROT_DIM) { 1f },
            "sin" to FloatArray(size * HostMath.ROT_DIM),
          )
        decide.run(size, feed)
        out["decide_L$size"] = decide.lastBuildMs
      }
    }
    return out
  }

  private fun zeroPicture() =
    HostMath.Picture(
      2,
      2,
      FloatArray(4 * HostMath.PATCH_DIM),
      FloatArray(4 * HostMath.POS_DIM),
      FloatArray(4 * HostMath.ROT_DIM) { 1f },
      FloatArray(4 * HostMath.ROT_DIM),
      HostMath.Rgb(ByteArray(32 * 32 * 3), 32, 32),
    )

  /**
   * Scores [text] against the region and its render. Throws [Refused] for a request the host
   * refuses. [dump], when given, receives each intermediate array (the app's check mode).
   */
  fun score(
    region: HostMath.Rgb,
    render: HostMath.Rgb,
    text: String,
    dump: ((String, Any) -> Unit)? = null,
  ): Score {
    val ms = LinkedHashMap<String, Double>()
    var build = 0.0
    var t = System.nanoTime()
    // a GPU build inside a step is taken out of that step's time and reported as buildMs
    fun lap(name: String, excludedMs: Double = 0.0) {
      val now = System.nanoTime()
      ms[name] = (ms[name] ?: 0.0) + (now - t) / 1e6 - excludedMs
      build += excludedMs
      t = now
    }
    val a = builder.picture(region)
    val b = builder.picture(render)
    lap("pictures")
    val req = builder.request(a, b, text)
    lap("tokens")
    val sizeA = towers.sizeFor(a.n)
    val sizeB = towers.sizeFor(b.n)
    val length = decide.sizeFor(req.tokens)
    if (sizeA == null || sizeB == null || length == null) {
      throw Refused(
        "too_large",
        "the pictures need ${a.n} and ${b.n} patches and the request ${req.tokens} tokens; the installed " +
          "files hold up to ${towers.maxSize} patches and ${decide.maxSize} tokens",
      )
    }
    val rowsA = towers.run(sizeA, builder.towerFeed(a, sizeA))
    lap("tower_region", towers.lastBuildMs)
    val rowsB = towers.run(sizeB, builder.towerFeed(b, sizeB))
    lap("tower_render", towers.lastBuildMs)
    val embeds = builder.embeds(req, rowsA, rowsB, length)
    val feed = builder.decideFeed(req, embeds, length)
    lap("rows_and_tables")
    val hidden = decide.run(length, feed)
    lap("decide", decide.lastBuildMs)
    val pooled = (req.tokens - 1) * HostMath.HIDDEN
    val (logits, p1) = HostMath.head(hidden, pooled, tables.scoreWeight)
    lap("head")
    if (dump != null) {
      dump("ids", req.ids)
      dump("region_u8", a.resized.bytes)
      dump("render_u8", b.resized.bytes)
      dump("region_patches", a.patches)
      dump("render_patches", b.patches)
      dump("region_pos", a.pos)
      dump("render_pos", b.pos)
      dump("region_cos", a.cos)
      dump("region_sin", a.sin)
      dump("render_cos", b.cos)
      dump("render_sin", b.sin)
      dump("region_rows", rowsA.copyOf(a.tokens * HostMath.HIDDEN))
      dump("render_rows", rowsB.copyOf(b.tokens * HostMath.HIDDEN))
      dump("embeds", embeds.copyOf(req.tokens * HostMath.HIDDEN))
      dump("mrope_cos", feed.getValue("cos"))
      dump("mrope_sin", feed.getValue("sin"))
      dump("valid", feed.getValue("valid"))
      dump("hidden_pooled", hidden.copyOfRange(pooled, pooled + HostMath.HIDDEN))
    }
    return Score(
      p1,
      logits,
      req.tokens,
      intArrayOf(a.n, b.n),
      intArrayOf(sizeA, sizeB),
      length,
      ms,
      build,
    )
  }

  override fun close() {
    runCatching { towers.close() }
    runCatching { decide.close() }
    runCatching { env.close() }
    tables.close()
  }
}
