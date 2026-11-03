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

package com.google.ai.edge.examples.offline_planner

import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.security.MessageDigest
import java.time.LocalDateTime
import java.util.concurrent.CountDownLatch
import org.json.JSONArray
import org.json.JSONObject

/**
 * The measuring tool (debug builds): runs a plan through the same functions the buttons call and
 * writes a report.
 *
 * ```
 * adb shell am start -n com.google.ai.edge.examples.offline_planner/.MainActivity \
 *   --es plan plans/example.json --es report example
 * ```
 *
 * The plan sits in the app's internal files dir (`adb shell run-as <package>` reaches it on a debug
 * build): `{"now": "2026-10-30T09:00", "verify_model":
 * true, "items": [...]}`. An item is `{"id", "request", "reset", "press", "run", "inject_reply",
 * "ready_timeout_ms", "rest_ms"}`: `reset` restores the example data at the plan's clock, `press`
 * taps Ask (after waiting for a ready engine), `run` taps Run after the reply, `inject_reply` puts
 * a given text through the reply's parse, check and display instead of the model. The report goes
 * to `reports/<report>.json` in the same dir, rewritten after every item.
 */
class AutoRun(private val activity: MainActivity, private val planPath: String, private val tag: String) {
  private val main = Handler(Looper.getMainLooper())

  @Volatile private var stopped = false
  private var thread: Thread? = null

  fun start() {
    thread = Thread({ runPlan() }, "planner-autorun").apply {
      isDaemon = true
      start()
    }
  }

  fun stop() {
    stopped = true
    thread?.interrupt()
  }

  private fun <T> onMain(block: () -> T): T {
    val latch = CountDownLatch(1)
    var result: Result<T>? = null
    main.post {
      result = runCatching(block)
      latch.countDown()
    }
    latch.await()
    return result!!.getOrThrow()
  }

  private fun waitUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
    val end = SystemClock.elapsedRealtime() + timeoutMs
    while (SystemClock.elapsedRealtime() < end && !stopped) {
      if (condition()) return true
      Thread.sleep(50)
    }
    return condition()
  }

  private fun runPlan() {
    // The app's internal files dir: on a debug build adb reaches it with `run-as`.
    val dir = activity.filesDir
    val reportFile = File(File(dir, "reports").apply { mkdirs() }, "$tag.json")
    val report = JSONObject()
    val items = JSONArray()
    try {
      val plan = JSONObject(File(dir, planPath).readText())
      plan.optString("now").takeIf { it.isNotEmpty() }?.let { onMain { activity.setFixedNow(LocalDateTime.parse(it)) } }
      val (airplane, online) = onMain { activity.networkState() }
      report
        .put("plan", planPath)
        .put("device", onMain { activity.deviceName() })
        .put("model_id", Build.MODEL)
        .put("android", Build.VERSION.RELEASE)
        .put("sdk", Build.VERSION.SDK_INT)
        .put("fingerprint", Build.FINGERPRINT)
        .put("litertlm", MainActivity.LITERT_LM_VERSION)
        .put("app", BuildConfig.VERSION_NAME)
        .put("now", plan.optString("now"))
        .put("airplane_mode", airplane)
        .put("online", online)
        .put("started", LocalDateTime.now().toString())
      val model = onMain { activity.modelFile() }
      report.put("model_file", model?.name).put("model_bytes", model?.takeIf { it.isFile }?.length() ?: -1)
      if (plan.optBoolean("verify_model") && model != null && model.isFile) report.put("model_sha256", sha256(model))
      report.put("items", items)
      val list = plan.getJSONArray("items")
      for (i in 0 until list.length()) {
        if (stopped) break
        val item = list.getJSONObject(i)
        items.put(runItem(item))
        write(reportFile, report.put("status", "running"))
        // A model that cannot load stops the plan (the no-model item itself records the screen first).
        val err = activity.loadError
        if (err != null && item.optBoolean("press") && !plan.optBoolean("continue_on_load_error")) {
          report.put("load_error", err)
          break
        }
      }
      report.put("backend", onMain { activity.backendName() })
      report.put("status", if (stopped) "stopped" else "done")
    } catch (t: Throwable) {
      Log.e(TAG, "plan failed", t)
      report.put("status", "error: ${t.javaClass.simpleName}: ${t.message}")
    }
    report.put("finished", LocalDateTime.now().toString())
    write(reportFile, report)
    Log.i(TAG, "AUTORUN ${report.optString("status")} report=${reportFile.absolutePath}")
  }

  private fun runItem(item: JSONObject): JSONObject {
    val out = JSONObject().put("id", item.optString("id"))
    val request = item.optString("request", "")
    out.put("request", request).put("thermal_before", onMain { activity.thermalStatus() })
    if (item.optBoolean("reset")) onMain { activity.resetState(activity.now()) }
    onMain { activity.setRequest(request) }
    when {
      item.has("inject_reply") -> {
        onMain { activity.showInjectedReply(item.getString("inject_reply")) }
        waitUntil(5_000) { activity.last?.done == true }
        out.put("injected", true)
        record(out)
      }
      item.optBoolean("press") -> {
        val readyTimeout = item.optLong("ready_timeout_ms", 300_000)
        if (request.isNotEmpty()) {
          out.put("ready", waitUntil(readyTimeout) { onMain { activity.engineReady() || activity.loadError != null } } && activity.loadError == null)
        }
        activity.loadError?.let { out.put("load_error", it) }
        val sent = onMain { activity.ask() }
        out.put("sent", sent)
        if (sent) {
          out.put("finished", waitUntil(REPLY_WAIT_MS) { activity.last?.done == true })
        }
        record(out)
        if (sent && item.optBoolean("run")) {
          out.put("run", JSONArray(onMain { activity.runActions() }))
        }
        if (sent) {
          // The next request's Engine, so that the next item starts like a tap on a ready screen.
          out.put("next_ready", waitUntil(readyTimeout) { onMain { activity.engineReady() } })
        }
      }
    }
    out.put("screen", JSONObject(onMain { activity.screenLines() }))
    out.put("state", onMain { activity.stateJson() })
    out.put("thermal_after", onMain { activity.thermalStatus() })
    out.put("rest_ms", item.optLong("rest_ms", 0))
    if (item.optLong("rest_ms", 0) > 0) Thread.sleep(item.getLong("rest_ms"))
    return out
  }

  /** The last request's reply, calls, checks and times. */
  private fun record(out: JSONObject) {
    val r = activity.last ?: return
    val j = r.toJson()
    for (k in j.keys()) if (k != "request" && k != "run") out.put(k, j.get(k))
  }

  private fun write(file: File, report: JSONObject) {
    val tmp = File(file.parentFile, file.name + ".tmp")
    tmp.writeText(report.toString(1))
    tmp.renameTo(file)
  }

  private fun sha256(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
      val buf = ByteArray(8 shl 20)
      while (true) {
        val n = input.read(buf)
        if (n < 0) break
        md.update(buf, 0, n)
      }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
  }

  companion object {
    private const val TAG = "OfflinePlanner"
    private const val REPLY_WAIT_MS = 180_000L
  }
}
