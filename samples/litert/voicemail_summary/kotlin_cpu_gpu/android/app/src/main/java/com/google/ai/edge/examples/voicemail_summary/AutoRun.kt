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

package com.google.ai.edge.examples.voicemail_summary

import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * The measuring tool: a plan run through the screen's own code, the same functions the buttons call
 * (MainActivity.selectBundled / openUri / toggleRecord / onSummarize), so a measured answer is a tapped answer.
 *
 *   adb shell am start -n com.google.ai.edge.examples.voicemail_summary/.MainActivity \
 *     --es plan plans/<name>.json --es report <tag>.json [--ez keep_frames true]
 *
 * Plan (in the app's files dir): {"items": [{"id", "voicemail": "<example id>" | "file": "<path under the files dir
 * or the external files dir>", "answer_in": "<language>" (the Answer in choice: its default instruction),
 * "instruction": "<text>" (else the field as it is), "path": "interleaved" | "two_stage" (else the language's),
 * "press": true | false, "seed": 0, "action": "record" (the Record button, for the permission path), "rest_ms": 1500,
 * "dump_input": true (the decoded input written to files/input_<tag>_<id>.f32, 16 kHz mono float32)}]}.
 * An item without "press" sets the screen only (the START screen of a recording).
 * Report: files/<tag>.json (status DONE or ENGINE_UNAVAILABLE, the device, airplane mode, the load record, every
 * item's record with the screen's lines), files/out_<tag>.f32 (the pressed items' speech one after the other, 24 kHz
 * float32, `pcm_offset` / `pcm_samples`), files/shots/<tag>_<id>.png (the app's own view after each item).
 * keep_frames: one corner pixel changes every 33 ms from then on (a still screen gives a screen recording no frames).
 */
class AutoRun(private val a: MainActivity, planName: String, private val reportName: String, private val keepFrames: Boolean) {
  private val plan: JSONObject? = runCatching { JSONObject(File(a.filesDir, planName).readText()) }.getOrNull()
  private val tag = reportName.removeSuffix(".json")
  private val records = JSONArray()
  private var index = 0
  private val ui = Handler(Looper.getMainLooper())
  private var waitingRecord: JSONObject? = null
  private var finished = false

  fun start() = ui.post {
    if (keepFrames) a.keepFrames(true)
    next()
  }

  private fun next() {
    val p = plan ?: return finish("FAILED", "the plan is not readable")
    val items = p.getJSONArray("items")
    if (index >= items.length()) return finish("DONE", null)
    val item = items.getJSONObject(index++)
    val rec = JSONObject().put("id", item.getString("id")).put("plan_item", item)
    when {
      item.has("voicemail") -> {
        val b = a.bundledById(item.getString("voicemail")) ?: return finish("FAILED", "no example ${item.getString("voicemail")}")
        a.selectBundled(b)
        act(item, rec)
      }
      item.has("file") -> {
        val f = listOfNotNull(a.filesDir, a.getExternalFilesDir(null)).map { File(it, item.getString("file")) }.firstOrNull { it.isFile }
          ?: return finish("FAILED", "no file ${item.getString("file")}")
        a.openUri(Uri.fromFile(f)) { ok -> ui.post { rec.put("input_usable", ok); act(item, rec) } }
      }
      else -> act(item, rec)
    }
  }

  private fun act(item: JSONObject, rec: JSONObject) {
    if (item.optBoolean("dump_input", false)) {
      // the decoded input as the model hears it: files/input_<tag>_<id>.f32 (16 kHz mono float32)
      a.currentInputPcm()?.let { pcm ->
        runCatching {
          val bb = ByteBuffer.allocate(pcm.size * 4).order(ByteOrder.LITTLE_ENDIAN)
          for (v in pcm) bb.putFloat(v)
          File(a.filesDir, "input_${tag}_${rec.getString("id")}.f32").writeBytes(bb.array())
          rec.put("input_samples", pcm.size)
        }.onFailure { rec.put("input_dump_error", it.toString()) }
      } ?: rec.put("input_samples", 0)
    }
    item.optString("answer_in").takeIf { it.isNotEmpty() }?.let { if (!a.setAnswerIn(it)) rec.put("answer_in_error", "no language $it") }
    item.optString("instruction").takeIf { it.isNotEmpty() }?.let { a.setInstruction(it) }
    val rest = item.optLong("rest_ms", 1500)
    when {
      item.optString("action") == "record" -> {
        waitingRecord = rec
        a.toggleRecord()
        // a refused permission answers through onRecordDenied; a granted one starts recording (stopped here after 3 s)
        ui.postDelayed({
          if (waitingRecord === rec) {
            waitingRecord = null
            if (a.isRecording()) a.toggleRecord() else rec.put("record_answer", "no permission result within 3 s")
            done(rec, rest)
          }
        }, 3000)
      }
      item.optBoolean("press", false) -> ui.postDelayed({
        a.onSummarize("autoplay", { r -> ui.post { rec.put("run", r ?: JSONObject.NULL); done(rec, rest) } },
          seed = item.optLong("seed", 0), id = "${tag}_${item.getString("id")}",
          pathOverride = item.optString("path").takeIf { it.isNotEmpty() })
      }, 300)
      else -> ui.postDelayed({ done(rec, rest) }, 500)
    }
  }

  fun onRecordDenied() = ui.post {
    val r = waitingRecord ?: return@post
    waitingRecord = null
    r.put("record_denied", true)
    done(r, 1500)
  }

  /** The screen as it is after the item: its lines, and a picture of it. */
  private fun done(rec: JSONObject, rest: Long) {
    ui.postDelayed({
      rec.put("screen", JSONObject().put("device", a.currentDeviceLine()).put("engine", a.currentEngineLine())
        .put("input", a.currentInputLabel()).put("notice", a.currentNotice()).put("answer", a.currentAnswer()))
      a.snapshot(File(a.filesDir, "shots/${tag}_${rec.getString("id")}.png"))
      records.put(rec)
      ui.postDelayed({ next() }, rest)
    }, 700)
  }

  /** The model could not load (missing files, memory): the report records the screen's sentence. */
  fun engineUnavailable(line: String) = ui.post {
    if (finished) return@post
    val rec = JSONObject().put("id", "engine").put("screen", JSONObject().put("device", a.currentDeviceLine()).put("engine", line)
      .put("input", a.currentInputLabel()).put("notice", a.currentNotice()))
    a.snapshot(File(a.filesDir, "shots/${tag}_engine.png"))
    records.put(rec)
    finish("ENGINE_UNAVAILABLE", line)
  }

  fun onInputSound(ns: Long) {
    // the input's playback start, for a recording's sound track (the run log has the answer's)
    runCatching { File(a.filesDir, "runs").mkdirs(); File(a.filesDir, "runs/input_sound.log").appendText("${System.currentTimeMillis()} $ns\n") }
  }

  private fun finish(status: String, error: String?) {
    if (finished) return
    finished = true
    // keep_frames stays on after the plan: the recording that follows it (the START plan of a take) needs the frames
    Thread {
      val report = JSONObject().put("tag", tag).put("status", status)
      if (error != null) report.put("error", error)
      report.put("litert_version", BuildConfig.LITERT_VERSION).put("npu_libs", BuildConfig.NPU_LIBS)
      report.put("device", JSONObject().put("model", Build.MODEL).put("android", Build.VERSION.RELEASE)
        .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else "?"))
      report.put("airplane_mode", a.airplaneMode())
      report.put("load", a.engineLoad())
      // the pressed items' speech, one after the other
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
