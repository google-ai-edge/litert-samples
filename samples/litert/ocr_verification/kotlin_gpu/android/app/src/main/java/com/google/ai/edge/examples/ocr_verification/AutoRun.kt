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

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import android.view.WindowManager
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * Runs the app from adb, through the same code as a finger (README, "Run it from a computer"):
 *
 *     adb shell am start -n <package>/.MainActivity --es run <what> [--ez dump true]
 *
 * `<what>` is one of
 * - `start`: the screen a person sees after opening the app (no photo, no result);
 * - `example:<file>`: an example page of the app (assets/examples/), as the Example button opens it;
 * - `photo:<name>`: a picture file the computer copied into the app's `files/inputs/`;
 * - `check:<case>[,<case>...]`: for each `files/check/<case>/` holding `block.png` (a region picture),
 *   `reference_render.png` (a render of the text made elsewhere), `text.txt` and `category.txt`
 *   (`text`, `formula` or `table`): scores the region with the given render, then draws the text
 *   with the app's renderer and scores the region with that render, and writes every graph input
 *   (raw little-endian arrays) and both scores under `files/check/<case>/out/`.
 *
 * With `dump`, an example or photo run also writes each line's region, render, text and graph
 * inputs under `files/check/run_<photo>/line<k>/`. `--ez check false` turns the model's check off
 * before the run (every line is accepted as the recognizer read it) and `--ez check true` turns it
 * on; the switch keeps the setting. Every run writes its numbers to `files/runs/` and to the log
 * (tag `OcrCheck`).
 */
object AutoRun {
  const val EXTRA = "run"
  const val EXTRA_DUMP = "dump"
  const val EXTRA_CHECK = "check"

  fun wants(intent: Intent?): Boolean = intent?.getStringExtra(EXTRA) != null

  /** A run from adb often finds the phone locked: show above the keyguard and keep the screen on. */
  fun wakeScreen(activity: Activity) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      activity.setShowWhenLocked(true)
      activity.setTurnScreenOn(true)
    }
    activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }

  fun start(activity: MainActivity, intent: Intent) {
    val what = intent.getStringExtra(EXTRA) ?: return
    val dump = intent.getBooleanExtra(EXTRA_DUMP, false)
    val check = if (intent.hasExtra(EXTRA_CHECK)) intent.getBooleanExtra(EXTRA_CHECK, true) else null
    // a re-created activity (rotation, process kept) must not run it again
    intent.removeExtra(EXTRA)
    intent.removeExtra(EXTRA_CHECK)
    Log.i(MainActivity.TAG, "AUTORUN $what dump=$dump check=$check")
    check?.let { activity.setCheck(it, scoreNow = false) }
    val done: () -> Unit = { Log.i(MainActivity.TAG, "AUTORUN done $what") }
    when {
      what == "start" -> {
        activity.resetToStart()
        done()
      }
      what.startsWith("example:") ->
        activity.openExample(what.removePrefix("example:"), dump = dump, then = done)
      what.startsWith("photo:") ->
        activity.openInput(what.removePrefix("photo:"), dump = dump, then = done)
      what.startsWith("check:") ->
        Check(activity, what.removePrefix("check:").split(',').filter { it.isNotBlank() }).run(0)
      else -> Log.w(MainActivity.TAG, "AUTORUN unknown '$what'")
    }
  }

  /** The check mode: the given render, then the app's own render, of each case. */
  private class Check(private val a: MainActivity, private val cases: List<String>) {
    fun run(i: Int) {
      if (i >= cases.size) {
        Log.i(MainActivity.TAG, "CHECK ALL DONE ${cases.size}")
        return
      }
      val case = cases[i]
      val dir = File(a.filesDir, "check/$case")
      val out = File(dir, "out")
      out.deleteRecursively()
      out.mkdirs()
      val v = a.verifierOrNull() ?: return fail(case, "model not loaded", i)
      a.workerExecutor().execute {
        val first = runCatching {
          val text = File(dir, "text.txt").readText(Charsets.UTF_8)
          val category = RenderPage.Category.valueOf(File(dir, "category.txt").readText().trim().uppercase())
          val region = Photo.rgbOfFile(File(dir, "block.png"))
          val given = Photo.rgbOfFile(File(dir, "reference_render.png"))
          val g = File(out, "given").apply { mkdirs() }
          writeArray(File(g, "region_input_u8.bin"), region.bytes)
          writeArray(File(g, "render_input_u8.bin"), given.bytes)
          val s = v.score(region, given, text) { name, arr -> writeArray(File(g, "$name.bin"), arr) }
          File(g, "score.json").writeText(scoreJson(s, region, given).toString(1))
          Log.i(MainActivity.TAG, "CHECK $case given prob %.6f T %d n %s".format(s.prob1, s.tokens, s.patches.toList()))
          Triple(text, category, region)
        }
        a.runOnUiThread {
          val (text, category, region) = first.getOrElse { e -> return@runOnUiThread fail(case, "given: $e", i) }
          a.rendererOrNull().render(text, category, region.width) { rendered ->
            val r = rendered.getOrElse { e -> return@render fail(case, "render: $e", i) }
            a.workerExecutor().execute {
              val second = runCatching {
                val app = File(out, "app").apply { mkdirs() }
                savePng(r.rgb, File(out, "app_render.png"))
                writeArray(File(app, "region_input_u8.bin"), region.bytes)
                writeArray(File(app, "render_input_u8.bin"), r.rgb.bytes)
                val s = v.score(region, r.rgb, text) { name, arr -> writeArray(File(app, "$name.bin"), arr) }
                File(app, "score.json").writeText(
                  scoreJson(s, region, r.rgb)
                    .put("render_ms", r.ms)
                    .put("render_css_width", r.cssWidth)
                    .put("katex_errors", JSONArray(r.errors))
                    .toString(1)
                )
                Log.i(MainActivity.TAG, "CHECK $case app prob %.6f T %d n %s render %dx%d".format(
                  s.prob1, s.tokens, s.patches.toList(), r.rgb.width, r.rgb.height))
              }
              a.runOnUiThread {
                second.onFailure { e -> fail(case, "app: $e", i) }.onSuccess {
                  Log.i(MainActivity.TAG, "CHECK $case done")
                  run(i + 1)
                }
              }
            }
          }
        }
      }
    }

    private fun fail(case: String, why: String, i: Int) {
      Log.e(MainActivity.TAG, "CHECK $case FAILED $why")
      run(i + 1)
    }
  }

  fun scoreJson(s: Score, region: HostMath.Rgb, render: HostMath.Rgb): JSONObject =
    JSONObject()
      .put("prob_1", s.prob1)
      .put("logits", JSONArray(s.logits.map { it.toDouble() }))
      .put("T", s.tokens)
      .put("n", JSONArray(s.patches.toList()))
      .put("tower_sizes", JSONArray(s.towerSizes.toList()))
      .put("decide_size", s.decideSize)
      .put("ms", JSONObject(s.ms as Map<*, *>))
      .put("build_ms", s.buildMs)
      .put("region_size", "${region.width}x${region.height}")
      .put("render_size", "${render.width}x${render.height}")
      .put("device", DeviceInfo.displayName())
      .put("mem_available_kb", DeviceInfo.memAvailableKb())
      .put("self_kb", JSONObject(DeviceInfo.selfStatusKb() as Map<*, *>))

  /** Raw little-endian bytes of a FloatArray, IntArray or ByteArray (the Mac compare reads them). */
  fun writeArray(file: File, arr: Any) {
    val bytes =
      when (arr) {
        is ByteArray -> arr
        is FloatArray ->
          ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(arr) }.array()
        is IntArray ->
          ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asIntBuffer().put(arr) }.array()
        else -> error("cannot write ${arr.javaClass}")
      }
    file.parentFile?.mkdirs()
    file.writeBytes(bytes)
  }

  /** An RGB picture as a PNG file (lossless: the same values come back). */
  fun savePng(rgb: HostMath.Rgb, file: File) {
    val px = IntArray(rgb.width * rgb.height)
    for (i in px.indices) {
      px[i] =
        (0xFF shl 24) or
          ((rgb.bytes[i * 3].toInt() and 0xFF) shl 16) or
          ((rgb.bytes[i * 3 + 1].toInt() and 0xFF) shl 8) or
          (rgb.bytes[i * 3 + 2].toInt() and 0xFF)
    }
    val bmp = Bitmap.createBitmap(px, rgb.width, rgb.height, Bitmap.Config.ARGB_8888)
    file.parentFile?.mkdirs()
    FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    bmp.recycle()
  }
}
