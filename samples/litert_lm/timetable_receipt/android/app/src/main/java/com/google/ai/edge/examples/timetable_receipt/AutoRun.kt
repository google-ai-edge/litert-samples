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

package com.google.ai.edge.examples.timetable_receipt

import android.app.ActivityManager
import android.content.Context
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
 * Start it with `adb shell am start -n <package>/.MainActivity --es plan <plan.json> --es report
 * <name>.json`; both paths are in the app's external files dir. A plan item may pick a picture
 * (`example` id, or `file` relative to the files dir), set `task`, `after`, `people` or
 * `question`, tap Take photo (`"action": "camera"`), and with `"press": true` tap Ask. `rest_ms`
 * waits after the item; `dump_input` keeps the picture the model was given (PNG) next to the report.
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
    val items = JSONArray()
    report.put("items", items)
    try {
      val plan = JSONObject(File(base, planPath).readText())
      settled.await()
      report.put("model", activity.statusText(ModelEngine.state))
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
        Task.parse(o.optString("task", ""))?.let { activity.setTask(it) }
        if (o.has("after")) activity.setAfter(o.getString("after"))
        if (o.has("people")) activity.setPeople(o.getInt("people"))
        if (o.has("question")) activity.setQuestion(o.getString("question"))
        if (o.optString("action") == "camera") {
          activity.takePhotoPressed()
          delay(o.optLong("camera_wait_ms", 2500))
        }
        if (o.optBoolean("press")) {
          rec.put("memory_before", availMemory(activity))
          val r = activity.ask()
          rec.put("answer", Records.toJson(r)).put("memory_after", availMemory(activity))
        }
        if (o.optBoolean("dump_input")) {
          activity.currentModelPng()?.let { File(base, "reports/${reportName.removeSuffix(".json")}_${rec.getString("id")}.png").writeBytes(it) }
        }
        rec.put("screen", JSONObject(activity.screenLines()))
        items.put(rec)
        write(reportFile, report)
        delay(o.optLong("rest_ms", 0))
      }
      report.put("done", true)
    } catch (t: Throwable) {
      report.put("error", t.toString())
    }
    report.put("airplane_mode", Settings.Global.getInt(activity.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0))
    write(reportFile, report)
    activity.autoRun = null
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
    .put("task", r.task.name)
    .put("picture", r.pictureName)
    .put("question", r.question)
    .put("text", r.text)
    .put("answer", r.answer)
    .put("values", JSONArray(r.values))
    .put("first_text_ms", r.firstTextMs ?: JSONObject.NULL)
    .put("answer_ms", r.answerMs)
    .put("error", r.error ?: JSONObject.NULL)

  fun save(context: Context, r: AnswerRecord) {
    val base = context.getExternalFilesDir(null) ?: context.filesDir
    val dir = File(base, "runs").apply { mkdirs() }
    File(dir, "run_${System.currentTimeMillis()}.json").writeText(
      toJson(r).put("airplane_mode", Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0)).toString(1)
    )
  }
}
