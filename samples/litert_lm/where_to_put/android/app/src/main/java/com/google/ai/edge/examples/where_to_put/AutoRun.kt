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

package com.google.ai.edge.examples.where_to_put

import android.app.ActivityManager
import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Build
import android.provider.Settings
import java.io.File
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * The measuring tool: runs a plan through the same functions the buttons call and writes a report.
 *
 * Start it with `adb shell am start -n <package>/.MainActivity --es plan <plan.json> --es report <name>.json`; both
 * paths are in the app's external files dir. A plan item may pick a picture (`example` id, or `file` relative to the
 * files dir), set the request (`request_id` of a bundled request, which fills the field as its suggestion button
 * does, or `request` text), tap Take photo (`"action": "camera"`), and with `"press": true` tap Show where. `rest_ms`
 * waits after the item; `dump_input` keeps the picture the model was given (PNG) next to the report. For a bundled
 * request the report says whether the point fell in the region where the item can go (`in_spot`).
 */
class AutoRun(private val activity: MainActivity, private val planPath: String, private val reportName: String) {
  private val settled = CompletableDeferred<Unit>()

  /** Called once the model load has ended, ready or not. */
  fun modelSettled() {
    settled.complete(Unit)
  }

  suspend fun run() {
    val base = activity.getExternalFilesDir(null) ?: activity.filesDir
    val reportFile = File(base, "reports/$reportName").apply { parentFile?.mkdirs(); delete() }
    val report = JSONObject()
      .put("plan", planPath)
      .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
      .put("android", Build.VERSION.RELEASE)
      .put("litert_lm", BuildConfig.LITERT_LM_VERSION)
      .put("thinking", ModelEngine.THINKING)
      .put("max_output_tokens", ModelEngine.maxOutputTokens)
    val items = JSONArray()
    report.put("items", items)
    var inSpot = 0
    var outSpot = 0
    var noMark = 0
    var tooLong = 0
    try {
      val plan = JSONObject(File(base, planPath).readText())
      settled.await()
      report.put("model", activity.statusText(ModelEngine.state))
      val examples = Examples.load(activity)
      val list = plan.getJSONArray("items")
      for (i in 0 until list.length()) {
        val o = list.getJSONObject(i)
        val rec = JSONObject().put("id", o.optString("id", "item$i"))
        when {
          o.has("example") -> activity.selectExample(o.getString("example"))
          o.has("file") -> {
            val f = File(base, o.getString("file"))
            activity.loadFile(f, o.optString("name", f.name))
          }
        }
        // The picture being read (the latest one, if a second read started meanwhile).
        while (activity.pendingPicture?.isActive == true) activity.pendingPicture?.join()
        var mask: String? = null
        if (o.has("request_id")) {
          val (e, r) = Examples.request(examples, o.getString("request_id"))
            ?: error("no bundled request ${o.getString("request_id")}")
          if (activity.currentExample?.id != e.id) error("request ${r.id} belongs to example ${e.id}")
          activity.setRequest(r.text)
          mask = r.spotMask
        } else if (o.has("request")) {
          activity.setRequest(o.getString("request"))
        }
        if (o.optString("action") == "camera") {
          activity.takePhotoPressed()
          delay(o.optLong("camera_wait_ms", 2500))
        }
        if (o.optBoolean("press")) {
          rec.put("memory_before", availMemory(activity))
          val r = activity.showWhere()
          val json = Records.toJson(r)
          val m = mask
          if (m != null) {
            val s = r.spot
            val hit = s != null && s.inPicture && withContext(Dispatchers.IO) { inMask(m, s) }
            json.put("spot_mask", m).put("in_spot", hit)
            if (hit) inSpot++ else if (r.marked) outSpot++ else noMark++
            if (r.tooLong) tooLong++
          }
          rec.put("answer", json).put("memory_after", availMemory(activity))
        }
        if (o.optBoolean("dump_input")) {
          activity.currentModelPng()?.let {
            File(base, "reports/${reportName.removeSuffix(".json")}_${rec.getString("id")}.png").writeBytes(it)
          }
        }
        rec.put("screen", JSONObject(activity.screenLines()))
        items.put(rec)
        write(reportFile, report)
        delay(o.optLong("rest_ms", 0))
      }
      report.put("in_spot", inSpot).put("marked_outside_spot", outSpot).put("no_mark", noMark)
        .put("no_mark_too_long", tooLong)
      report.put("done", true)
    } catch (t: Throwable) {
      report.put("error", t.toString())
    }
    report.put("airplane_mode", Settings.Global.getInt(activity.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0))
    write(reportFile, report)
    activity.autoRun = null
  }

  /** True when the mask's pixel under the point is white (the region where the item can go). */
  private fun inMask(asset: String, s: Spot): Boolean {
    val bitmap = activity.assets.open(asset).use { BitmapFactory.decodeStream(it) }
      ?: error("cannot decode $asset")
    try {
      check(bitmap.width == Pointing.MASK_SIDE && bitmap.height == Pointing.MASK_SIDE) {
        "$asset is ${bitmap.width} x ${bitmap.height}, not ${Pointing.MASK_SIDE} x ${Pointing.MASK_SIDE}"
      }
      val (px, py) = Pointing.maskPixel(s)
      return Color.red(bitmap.getPixel(px, py)) > 127
    } finally {
      bitmap.recycle()
    }
  }

  private suspend fun write(file: File, json: JSONObject) = withContext(Dispatchers.IO) {
    val tmp = File(file.path + ".tmp")
    tmp.writeText(json.toString(1))
    tmp.renameTo(file)
  }

  private fun availMemory(context: Context): Long {
    val m = ActivityManager.MemoryInfo()
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(m)
    return m.availMem
  }
}

/** Every answer, tapped or planned, leaves a record in the app's `files/runs/`. */
object Records {
  fun toJson(r: AnswerRecord): JSONObject = JSONObject()
    .put("picture", r.pictureName)
    .put("request", r.request)
    .put("instruction", r.instruction)
    .put("thinking", r.thinking)
    .put("text", r.text)
    .put("thought", r.thought)
    .put("point", r.spot?.let { JSONArray().put(it.x).put(it.y) } ?: JSONObject.NULL)
    .put("point_read_from", r.spot?.how ?: JSONObject.NULL)
    .put("marked", r.marked)
    .put("too_long", r.tooLong)
    .put("pieces", r.pieces)
    .put("max_output_tokens", r.maxOutputTokens)
    .put("first_text_ms", r.firstTextMs ?: JSONObject.NULL)
    .put("answer_ms", r.answerMs ?: JSONObject.NULL)
    .put("mark_ms", r.markMs ?: JSONObject.NULL)
    .put("total_ms", r.totalMs)
    .put("error", r.error ?: JSONObject.NULL)

  fun save(context: Context, r: AnswerRecord) {
    val base = context.getExternalFilesDir(null) ?: context.filesDir
    val dir = File(base, "runs").apply { mkdirs() }
    File(dir, "run_${System.currentTimeMillis()}.json").writeText(
      toJson(r).put("airplane_mode", Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0)).toString(1)
    )
  }
}
