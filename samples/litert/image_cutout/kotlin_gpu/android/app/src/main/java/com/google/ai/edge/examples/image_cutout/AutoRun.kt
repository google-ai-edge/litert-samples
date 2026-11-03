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

package com.google.ai.edge.examples.image_cutout

import android.os.Build
import android.os.Process
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug builds only: runs a plan of requests through the same path as the Run button (photo decode,
 * [MainActivity.run], the screen) and writes what each produced, for the on-device checks, the
 * timing and the failure paths.
 *
 * ```
 * adb shell am start -n com.google.ai.edge.examples.image_cutout/.MainActivity \
 *   --es autorun plans/<plan>.json --es out runs/<name>
 * ```
 *
 * Plan (paths under the app's files/):
 * ```
 * {"form": "gpu-fp16acc32", "reserve_mb": 800, "mem_check": true, "gpu_cache": true,
 *  "threads": 6, "save_arrays": true, "repeat": 1,
 *  "rows": [{"id": ..., "image": "<path under files/>" or "", "task": "<package task>",
 *            "query": ..., "instruction": "<exact instruction, optional>"}]}
 * ```
 *
 * Output: <out>/header.json, <out>/<id>[.<rep>].json (+ .inst<k>.alpha.f32 / .logits.f32 /
 * .tseg.f32, little-endian float32), <out>/DONE.json; logcat tag WeVisOneAutoRun: START / ROW /
 * DONE lines. A file `<out>/STOP` ends the plan after the current request.
 */
class AutoRun(
  private val activity: MainActivity,
  private val planFile: File,
  private val outDir: File,
) {
  fun start() {
    Thread(
        {
          try {
            runAll()
          } catch (e: Throwable) {
            Log.e(TAG, "WEVISONE_AUTORUN FAILED", e)
            runCatching {
              File(outDir, "DONE.json")
                .writeText(
                  JSONObject().put("status", "FAILED").put("error", e.toString()).toString(1)
                )
            }
          }
        },
        "WeVisOneAutoRun",
      )
      .start()
  }

  private fun runAll() {
    val plan = JSONObject(planFile.readText())
    outDir.mkdirs()
    File(outDir, "DONE.json").delete()
    val form = decoderFormOf(plan.optString("form", "")) ?: MainActivity.DEFAULT_FORM
    val options =
      RunOptions(
        reserveMb = plan.optLong("reserve_mb", WeVisOneEngine.DEFAULT_RESERVE_MB),
        memoryCheck = plan.optBoolean("mem_check", true),
        gpuProgramCache = plan.optBoolean("gpu_cache", true),
        cpuThreads = plan.optInt("threads", 6),
      )
    val saveArrays = plan.optBoolean("save_arrays", true)
    val repeat = plan.optInt("repeat", 1)
    val header =
      JSONObject()
        .put("plan", planFile.name)
        .put("form", form.key)
        .put("form_label", form.label)
        .put(
          "options",
          JSONObject()
            .put("reserve_mb", options.reserveMb)
            .put("mem_check", options.memoryCheck)
            .put("gpu_cache", options.gpuProgramCache)
            .put("threads", options.cpuThreads),
        )
        .put("device", DeviceInfo.displayName())
        .put("model", Build.MODEL)
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("fingerprint", Build.FINGERPRINT)
        .put("litert", BuildConfig.LITERT_VERSION)
        .put("pid", Process.myPid())
        .put("mem_total_kb", DeviceInfo.memTotalKb())
        .put("mem_available_kb", DeviceInfo.memAvailableKb())
        .put("thermal", DeviceInfo.thermalStatus(activity))
        .put("start_ms", System.currentTimeMillis())
    File(outDir, "header.json").writeText(header.toString(1))
    Log.i(
      TAG,
      "WEVISONE_AUTORUN START plan=${planFile.name} form=${form.key} pid=${Process.myPid()}",
    )
    val rows = plan.getJSONArray("rows")
    val summary = JSONArray()
    loop@ for (r in 0 until rows.length()) {
      val row = rows.getJSONObject(r)
      for (rep in 0 until repeat) {
        if (File(outDir, "STOP").exists()) break@loop
        val rec = runRow(row, rep, form, options, saveArrays)
        summary.put(
          JSONObject()
            .put("id", row.getString("id"))
            .put("rep", rep)
            .put("outcome", rec.optString("outcome"))
            .put("total_ms", rec.optJSONObject("timing")?.opt("total_ms"))
        )
        Log.i(
          TAG,
          "WEVISONE_AUTORUN ROW ${row.getString("id")} rep=$rep " +
            "outcome=${rec.optString("outcome")} " +
            "total_ms=${rec.optJSONObject("timing")?.opt("total_ms")}",
        )
      }
    }
    val done =
      JSONObject()
        .put("status", "DONE")
        .put("rows", summary)
        .put("pid", Process.myPid())
        .put("end_ms", System.currentTimeMillis())
        .put("self_status_kb", JSONObject(DeviceInfo.selfStatusKb()))
    File(outDir, "DONE.json").writeText(done.toString(1))
    Log.i(TAG, "WEVISONE_AUTORUN DONE rows=${summary.length()} pid=${Process.myPid()}")
  }

  private fun runRow(
    row: JSONObject,
    rep: Int,
    form: DecoderForm,
    options: RunOptions,
    saveArrays: Boolean,
  ): JSONObject {
    val id = row.getString("id")
    val stem = if (rep == 0) id else "$id.$rep"
    val task = taskOf(row.optString("task", "matting"))
    val query = row.optString("query", "")
    val instruction = row.optString("instruction", "").ifEmpty { null }
    val image = row.optString("image", "")
    val rec =
      JSONObject()
        .put("id", id)
        .put("rep", rep)
        .put("task", task.packageTask)
        .put("query", query)
        .put("instruction", instruction ?: Prompts.instruction(task, query))
        .put("image", image)
        .put("form", form.key)
        .put("thermal_before", DeviceInfo.thermalStatus(activity))
        .put("mem_available_kb_before", DeviceInfo.memAvailableKb())
        .put("cpuset", DeviceInfo.cpuset())
        .put("start_ms", System.currentTimeMillis())
    // the photo through the screen's decoder
    var photo: Photo? = null
    if (image.isNotEmpty()) {
      val t = System.nanoTime()
      activity.onWorker {
        photo =
          runCatching { Photo.fromFile(File(activity.filesDir, image)) }
            .onFailure { rec.put("decode_error", it.toString()) }
            .getOrNull()
      }
      rec.put("decode_ms", (System.nanoTime() - t) / 1e6)
    }
    onUi {
      activity.setPhoto(photo)
      activity.setInputs(task, query, form)
    }
    photo?.let {
      rec.put("width", it.width).put("height", it.height).put("pixel_sha256", it.pixelSha256)
    }
    var outcome: Outcome? = null
    val latch = CountDownLatch(1)
    activity.runOnUiThread {
      activity.run(task, query, instruction, form, options) {
        outcome = it
        latch.countDown()
      }
    }
    latch.await()
    rec
      .put("end_ms", System.currentTimeMillis())
      .put("thermal_after", DeviceInfo.thermalStatus(activity))
      .put("pid", Process.myPid())
      .put("self_status_kb", JSONObject(DeviceInfo.selfStatusKb()))
    val prediction =
      when (val o = outcome) {
        is Outcome.Done -> {
          rec.put("outcome", "done")
          o.prediction
        }
        is Outcome.Failed -> {
          rec.put("outcome", o.kind).put("message", o.message)
          if (o.kind == MainActivity.FAIL_NO_TARGET) activity.failedWith else null
        }
        null -> null
      }
    if (prediction != null) {
      rec
        .put("prompt_ids", JSONArray(prediction.promptIds.toList()))
        .put("n_prompt", prediction.promptIds.size)
        .put("new_ids", JSONArray(prediction.newIds.toList()))
        .put("n_new", prediction.newIds.size)
        .put("response", prediction.response)
        .put("stopped", prediction.stopped)
        .put("seg_boxes", JSONArray(prediction.answer.segBoxes.map { JSONArray(it.toList()) }))
        .put(
          "shapes",
          JSONArray(
            prediction.answer.shapes.map {
              JSONObject()
                .put("label", it.label)
                .put("kind", it.kind)
                .put("bins", JSONArray(it.bins.toList()))
            }
          ),
        )
        .put("stated_count", prediction.answer.statedCount ?: JSONObject.NULL)
        .put("timing", json(prediction.timing))
        .put("host_outside_engine_ms", JSONObject(activity.lastPrep as Map<*, *>))
        .put("memory", JSONArray(prediction.memory.map { json(it) }))
        .put("instances", prediction.instances.size)
      if (saveArrays) {
        for ((k, inst) in prediction.instances.withIndex()) {
          writeFloats(File(outDir, "$stem.inst$k.alpha.f32"), inst.alpha)
          writeFloats(File(outDir, "$stem.inst$k.logits.f32"), inst.logits)
          writeFloats(File(outDir, "$stem.inst$k.tseg.f32"), inst.tSeg)
        }
      }
    }
    File(outDir, "$stem.json").writeText(rec.toString(1))
    return rec
  }

  private fun onUi(block: () -> Unit) {
    val latch = CountDownLatch(1)
    activity.runOnUiThread {
      try {
        block()
      } finally {
        latch.countDown()
      }
    }
    latch.await()
  }

  private fun json(map: Map<String, Any?>): JSONObject {
    val o = JSONObject()
    for ((k, v) in map) {
      o.put(
        k,
        when (v) {
          null -> JSONObject.NULL
          is List<*> -> JSONArray(v.map { it ?: JSONObject.NULL })
          is Map<*, *> -> JSONObject(v as Map<*, *>)
          else -> v
        },
      )
    }
    return o
  }

  private fun writeFloats(file: File, data: FloatArray) {
    val buf = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asFloatBuffer().put(data)
    FileOutputStream(file).use { it.write(buf.array()) }
  }

  companion object {
    const val TAG = "WeVisOneAutoRun"
  }
}
