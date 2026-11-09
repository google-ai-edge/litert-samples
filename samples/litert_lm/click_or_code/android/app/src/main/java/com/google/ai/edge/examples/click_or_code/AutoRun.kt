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

package com.google.ai.edge.examples.click_or_code

import android.content.res.AssetManager
import android.util.Log
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Debug builds: a plan of steps run through the same code as the screen (the screenshot shown, the
 * request typed, the Decide button's path), one result line each, for the checks in the README.
 *
 * The plan is a JSON file in the app's external files dir:
 * ```
 * {"rows": [{"tag": "t1", "example": "files", "request": "Open the photo of the sailboat."},
 *           {"tag": "t2", "image": "plan/screen.png", "request": "Turn off Bluetooth."}, ...]}
 * ```
 *
 * `example` is a bundled example's id, `image` a path relative to the external files dir; absent:
 * the screen's picture stays. Optional per row: `render` (true: the line also gets the prompt as the
 * chat template rendered it), `cancel_after_ms` (stop the step after that long), `no_wait` (start
 * right after the previous step, before the next Conversation is ready; the default waits for it),
 * `need_bytes` (the memory the check asks for, from this row on), `start` (true: show the picture and
 * the request, clear the last answer, run nothing: the screen a take starts from). Results go to
 * `runs/<plan name>.jsonl` in the external files dir, then `AUTORUN_DONE` is logged.
 */
class AutoRun(
  private val vm: StepViewModel,
  private val root: File,
  private val assets: AssetManager,
) {
  fun start(scope: CoroutineScope, planName: String) {
    scope.launch {
      val planFile = File(root, planName)
      val out = File(root, "runs/${planFile.nameWithoutExtension}.jsonl")
      try {
        out.parentFile?.mkdirs()
        val rows = JSONObject(planFile.readText()).getJSONArray("rows")
        Log.i(TAG, "AUTORUN_START plan=$planName rows=${rows.length()} out=${out.path}")
        vm.ui.first { it.model is ModelState.Ready || it.model is ModelState.Error }
        for (i in 0 until rows.length()) {
          val row = rows.getJSONObject(i)
          val record = JSONObject().put("tag", row.optString("tag", "row$i")).put("index", i)
          if (row.has("need_bytes")) vm.setNeededMemory(row.getLong("need_bytes")).join()
          vm.dismissMessage()
          val picture = picture(row, record)
          if (record.has("error")) {
            out.appendText(record.toString() + "\n")
            Log.i(TAG, "AUTORUN_ROW ${record.optString("tag")} error=${record.opt("error")}")
            continue
          }
          if (picture != null) {
            val (bytes, name, chips) = picture
            vm.setPhoto(bytes, name, chips).join()
            vm.ui.value.message?.let { msg ->
              record.put("error", msg)
              vm.dismissMessage()
              out.appendText(record.toString() + "\n")
              Log.i(TAG, "AUTORUN_ROW ${record.optString("tag")} error=$msg")
              continue
            }
          }
          vm.setRequest(row.optString("request", ""))
          if (row.optBoolean("start", false)) {
            record.put("start", true)
            out.appendText(record.toString() + "\n")
            Log.i(TAG, "AUTORUN_ROW ${record.optString("tag")} start")
            continue
          }
          if (!row.optBoolean("no_wait", false)) {
            vm.ui.first { it.engine is EngineState.Ready || it.engine is EngineState.Failed }
          }
          val job =
            vm.decideOnce(
              record,
              renderPrompt = row.optBoolean("render", false),
              cancelAfterMs =
                if (row.has("cancel_after_ms")) row.getLong("cancel_after_ms") else null,
            )
          if (job == null) {
            record.put("no_step", vm.ui.value.message ?: "the model is not ready (${vm.ui.value.engine})")
          } else {
            job.join()
          }
          out.appendText(record.toString() + "\n")
          Log.i(
            TAG,
            "AUTORUN_ROW ${record.optString("tag")} total_ms=${record.opt("total_ms")} " +
              "error=${record.opt("error")}",
          )
          delay(300)
        }
        Log.i(TAG, "AUTORUN_DONE rows=${rows.length()} out=${out.path}")
      } catch (e: Exception) {
        Log.e(TAG, "AUTORUN_FAILED ${e.javaClass.simpleName}: ${e.message}")
      }
    }
  }

  /** The row's picture: bytes, name and the requests to offer; null when the row names none. */
  private fun picture(row: JSONObject, record: JSONObject): Triple<ByteArray, String, List<String>>? {
    if (row.has("example")) {
      val id = row.getString("example")
      record.put("example", id)
      val example = vm.ui.value.examples.firstOrNull { it.id == id }
      if (example == null) {
        record.put("error", "no example $id")
        return null
      }
      return Triple(
        assets.open("examples/${example.file}").use { it.readBytes() },
        example.name,
        example.requests,
      )
    }
    if (row.has("image")) {
      val image = File(root, row.getString("image"))
      record.put("image", row.getString("image"))
      val bytes = runCatching { image.readBytes() }.getOrNull()
      if (bytes == null) {
        record.put("error", "cannot read ${image.path}")
        return null
      }
      return Triple(bytes, image.name, emptyList())
    }
    return null
  }

  private companion object {
    const val TAG = StepViewModel.TAG
  }
}
