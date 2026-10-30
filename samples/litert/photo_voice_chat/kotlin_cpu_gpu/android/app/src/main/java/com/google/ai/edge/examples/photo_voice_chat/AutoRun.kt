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

package com.google.ai.edge.examples.photo_voice_chat

import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray
import org.json.JSONObject

/**
 * The measuring tool: a plan run through the screen's own code, the same functions the buttons call
 * (MainActivity.selectExample / openPhotoUri / askExample / askRecording / askTyped / micDown / micUp / takePhoto /
 * newConversation), so a measured answer is a tapped answer.
 *
 *   adb shell am start -n com.google.ai.edge.examples.photo_voice_chat/.MainActivity \
 *     --es plan plans/<name>.json --es report <tag>.json [--ez keep_frames true]
 *
 * Plan (in the app's files dir): {"items": [{"id", one of
 *   "photo": "<example id>" | "photo_file": "<path under the files dirs>"   choose the photo (a new conversation)
 *   "ask": "<example question id>" | "ask_wav": "<path>" | "ask_text": "<question>"   ask (a recording plays first)
 *   "action": "record" (hold the button for "hold_ms", the permission path) | "camera" (the camera app; the driver
 *             presses BACK or the shutter) | "new_conversation" | "engine" (opened again with "<part>_threads": n),
 * and for an ask: "instruction" (else the app's), "history": "state" | "text", "seed": 0, "audio_temperature",
 * "fixed_order": true (each frame's maskgit order 0..11), "dump": true (the vision rows, the audio rows and the ids to
 * files/dumps/<tag>_<id>_*), "rest_ms": 1500}]}.
 * Report: files/<tag>.json (status DONE or ENGINE_UNAVAILABLE, the device, airplane mode, the load record, every
 * item's record with the screen's lines), files/out_<tag>.f32 (the answers' speech one after the other, 24 kHz
 * float32, `pcm_offset` / `pcm_samples`), files/shots/<tag>_<id>.png (the app's own view after each item).
 * keep_frames: one pixel changes every 33 ms from then on (a still screen gives a screen recording no frames).
 */
class AutoRun(private val a: MainActivity, planName: String, private val reportName: String, private val keepFrames: Boolean) {
  private val plan: JSONObject? = runCatching { JSONObject(File(a.filesDir, planName).readText()) }.getOrNull()
  private val tag = reportName.removeSuffix(".json")
  private val records = JSONArray()
  private var index = 0
  private val ui = Handler(Looper.getMainLooper())
  private var waiting: JSONObject? = null
  private var finished = false

  fun start() = ui.post {
    if (keepFrames) a.keepFrames(true)
    next()
  }

  private fun file(path: String): File? = listOfNotNull(a.filesDir, a.getExternalFilesDir(null)).map { File(it, path) }.firstOrNull { it.isFile }

  private fun next() {
    val p = plan ?: return finish("FAILED", "the plan is not readable")
    val items = p.getJSONArray("items")
    if (index >= items.length()) return finish("DONE", null)
    val item = items.getJSONObject(index++)
    val rec = JSONObject().put("id", item.getString("id")).put("plan_item", item)
    val rest = item.optLong("rest_ms", 1500)
    item.optString("instruction").takeIf { item.has("instruction") }?.let { a.instruction = it }
    val opts = MainActivity.AskOptions(
      seed = item.optLong("seed", 0),
      history = item.optString("history", "state"),
      sampling = if (item.has("audio_temperature")) Sampling(audioTemperature = item.getDouble("audio_temperature").toFloat()) else Sampling(),
      fixedOrder = item.optBoolean("fixed_order", false),
      dump = if (item.optBoolean("dump", false)) "${tag}_${item.getString("id")}" else null,
      id = "${tag}_${item.getString("id")}",
    )
    val answered: (JSONObject?) -> Unit = { r -> ui.post { rec.put("run", r ?: JSONObject.NULL); done(rec, rest) } }
    when {
      item.has("photo") -> {
        val e = a.exampleById(item.getString("photo")) ?: return finish("FAILED", "no example ${item.getString("photo")}")
        a.selectExample(e) { ok -> ui.post { rec.put("photo_ready", ok); done(rec, rest) } }
      }
      item.has("photo_file") -> {
        val f = file(item.getString("photo_file")) ?: return finish("FAILED", "no file ${item.getString("photo_file")}")
        a.openPhotoUri(Uri.fromFile(f), { ok -> ui.post { rec.put("photo_ready", ok); done(rec, rest) } })
      }
      item.has("ask") -> {
        val q = a.questionById(item.getString("ask")) ?: return finish("FAILED", "no question ${item.getString("ask")}")
        ui.postDelayed({ a.askExample(q, answered, opts) }, 300)
      }
      item.has("ask_wav") -> {
        val f = file(item.getString("ask_wav")) ?: return finish("FAILED", "no file ${item.getString("ask_wav")}")
        ui.postDelayed({ a.askWavFile(f, item.optString("label").ifEmpty { null }, answered, opts) }, 300)
      }
      item.has("ask_text") -> ui.postDelayed({ a.askTyped(item.getString("ask_text"), answered, opts) }, 300)
      item.optString("action") == "record" -> {
        waiting = rec
        a.micDown()
        // a refused permission answers through onRecordDenied; a granted one records for hold_ms, then lets go
        ui.postDelayed({
          if (waiting === rec) {
            waiting = null
            if (a.isRecording()) {
              a.micUp()
              rec.put("recorded", true)
            } else rec.put("record_answer", "no recording and no permission result")
            done(rec, rest)
          }
        }, item.optLong("hold_ms", 2500))
      }
      item.optString("action") == "camera" -> {
        waiting = rec
        a.takePhoto()
        ui.postDelayed({
          if (waiting === rec) {
            waiting = null
            rec.put("camera_answer", "no result within ${item.optLong("wait_ms", 30000)} ms")
            done(rec, rest)
          }
        }, item.optLong("wait_ms", 30000))
      }
      item.optString("action") == "new_conversation" -> a.newConversation { ok -> ui.post { rec.put("restarted", ok); done(rec, rest) } }
      item.optString("action") == "engine" -> {
        // the CPU threads of each part ("lm_threads", "depth_threads", "frame_threads", "codec_threads", "audio_threads",
        // "vision_threads"; 4 when not given): the engine opened again, then a photo must be chosen again
        fun t(k: String) = Accel("cpu", threads = item.optInt(k, 4))
        // "lm": "gpu" = the language model on the GPU ("lm_precision", default = the runtime's), one copy of the weights
        // for its seven signatures and the compiled programs kept in the app's cache (Accel.sharedWeights)
        val lm = if (item.optString("lm") == "gpu") Accel("gpu", item.optString("lm_precision", "default"), sharedWeights = true) else t("lm_threads")
        val config = Lfm3OmniEngine.Config(lm = lm, depth = t("depth_threads"), frame = t("frame_threads"),
          codec = t("codec_threads"), audio = t("audio_threads"), vision = t("vision_threads"))
        a.reopenEngine(config) { ok -> ui.post { rec.put("engine_reopened", ok).put("config", config.toString()); rec.put("load", a.engineLoad()); done(rec, rest) } }
      }
      else -> ui.postDelayed({ done(rec, rest) }, 500)
    }
  }

  fun onRecordDenied() = ui.post {
    val r = waiting ?: return@post
    waiting = null
    r.put("record_denied", true)
    done(r, 1500)
  }

  fun onCameraResult(ok: Boolean) = ui.post {
    val r = waiting ?: return@post
    waiting = null
    r.put("camera_photo", ok)
    done(r, 1500)
  }

  fun onQuestionSound(ns: Long) {
    // a question recording's playback start, for a screen recording's sound track (the run records have the answers')
    runCatching { File(a.filesDir, "runs").mkdirs(); File(a.filesDir, "runs/question_sound.log").appendText("${System.currentTimeMillis()} $ns\n") }
  }

  /** The screen as it is after the item: its lines, and a picture of it. */
  private fun done(rec: JSONObject, rest: Long) {
    ui.postDelayed({
      rec.put("screen", JSONObject().put("device", a.currentDeviceLine()).put("engine", a.currentEngineLine())
        .put("photo", a.currentPhotoStatus()).put("notice", a.currentNotice()).put("turns", a.currentTurns()))
      a.snapshot(File(a.filesDir, "shots/${tag}_${rec.getString("id")}.png"))
      records.put(rec)
      ui.postDelayed({ next() }, rest)
    }, 700)
  }

  /** The model could not load (missing files, memory): the report records the screen's sentence. */
  fun engineUnavailable(line: String) = ui.post {
    if (finished) return@post
    val rec = JSONObject().put("id", "engine").put("screen", JSONObject().put("device", a.currentDeviceLine()).put("engine", line)
      .put("photo", a.currentPhotoStatus()).put("notice", a.currentNotice()))
    a.snapshot(File(a.filesDir, "shots/${tag}_engine.png"))
    records.put(rec)
    finish("ENGINE_UNAVAILABLE", line)
  }

  private fun finish(status: String, error: String?) {
    if (finished) return
    finished = true
    // keep_frames stays on after the plan: a screen recording after it (a START plan) needs the frames
    Thread {
      val report = JSONObject().put("tag", tag).put("status", status)
      if (error != null) report.put("error", error)
      report.put("litert_version", BuildConfig.LITERT_VERSION)
      report.put("device", JSONObject().put("model", Build.MODEL).put("android", Build.VERSION.RELEASE)
        .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else "?"))
      report.put("airplane_mode", a.airplaneMode())
      report.put("load", a.engineLoad())
      // the answers' speech, one after the other
      val out = File(a.filesDir, "out_$tag.f32")
      var offset = 0L
      val os = FileOutputStream(out)
      try {
        for (i in 0 until records.length()) {
          val run = records.getJSONObject(i).optJSONObject("run") ?: continue
          val f = run.optString("pcm_file").takeIf { it.isNotEmpty() }?.let { File(a.filesDir, it) } ?: continue
          if (!f.isFile) continue
          val bytes = f.readBytes()
          os.write(bytes)
          run.put("pcm_offset", offset)
          offset += bytes.size / 4
        }
      } finally {
        os.close()
      }
      report.put("items", records).put("out_bytes", out.length())
      val part = File(a.filesDir, "$reportName.partial")
      part.writeText(report.toString(1))
      part.renameTo(File(a.filesDir, reportName))
    }.start()
  }
}
