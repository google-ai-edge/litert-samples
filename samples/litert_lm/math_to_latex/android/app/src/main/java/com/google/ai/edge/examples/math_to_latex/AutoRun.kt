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

package com.google.ai.edge.examples.math_to_latex

import android.util.Log
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug builds: a plan of reads run through the same code as the screen (the photo shown, the box
 * drawn, the Read button's path), one result line each, for the checks in the README.
 *
 * The plan is a JSON file in the app's external files dir:
 * ```
 * {"rows": [{"tag": "t1", "image": "plan/page.png", "box": [x0, y0, x1, y1],
 *            "mode": "EQUATION"}, ...]}
 * ```
 *
 * `image` is relative to the external files dir; `box` is in the picture's pixels (after its EXIF
 * turn), or absent for the whole picture; `mode` is EQUATION or TEXT. Optional per row: `render`
 * (true: the line also gets the prompt as the chat template rendered it), `cancel_after_ms` (stop
 * the read after that long), `no_wait` (start right after the previous read, before the next Engine
 * is ready; the default waits for it), `need_bytes` (the memory the check asks for, from this row
 * on). Results go to `runs/<plan name>.jsonl` beside the plan, then `AUTORUN_DONE` is logged.
 */
class AutoRun(private val vm: MathViewModel, private val root: File) {
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
          vm.setMode(Mode.valueOf(row.optString("mode", "EQUATION")))
          val image = File(root, row.getString("image"))
          record.put("image", row.getString("image"))
          val bytes = runCatching { image.readBytes() }.getOrNull()
          if (bytes == null) {
            record.put("error", "cannot read ${image.path}")
            out.appendText(record.toString() + "\n")
            continue
          }
          vm.dismissMessage()
          vm.setPhoto(bytes, image.name).join()
          vm.ui.value.message?.let { msg ->
            record.put("error", msg)
            vm.dismissMessage()
            out.appendText(record.toString() + "\n")
            Log.i(TAG, "AUTORUN_ROW ${record.optString("tag")} error=$msg")
            continue
          }
          val box = row.optJSONArray("box")?.let { a -> IntArray(4) { a.getInt(it) } }
          record.put("box", box?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
          if (!row.optBoolean("no_wait", false)) {
            vm.ui.first { it.reader is ReaderState.Ready || it.reader is ReaderState.Failed }
          }
          val job =
            vm.readOnce(
              record,
              pixelBox = box,
              renderPrompt = row.optBoolean("render", false),
              cancelAfterMs =
                if (row.has("cancel_after_ms")) row.getLong("cancel_after_ms") else null,
            )
          if (job == null) {
            record.put("error", "the reader is not ready (${vm.ui.value.reader})")
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

  private companion object {
    const val TAG = MathViewModel.TAG
  }
}
