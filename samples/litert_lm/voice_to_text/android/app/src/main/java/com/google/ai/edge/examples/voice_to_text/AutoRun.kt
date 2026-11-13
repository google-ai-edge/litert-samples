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

package com.google.ai.edge.examples.voice_to_text

import android.util.Log
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Debug builds: a plan of WAV files run through the path a recording takes once Stop is tapped
 * (the checks, the WAV the model gets, the model), one result line each, for the checks in the
 * README.
 *
 * The plan is a JSON file in the app's external files dir:
 * ```
 * {"rows": [{"tag": "t1", "wav": "plan/clip.wav", "task": "TRANSCRIPT", "keywords": ""}, ...]}
 * ```
 *
 * `wav` is relative to the external files dir: 16 kHz mono, 16-bit PCM or 32-bit float. Optional
 * per row: `task` (TRANSCRIPT or ENGLISH), `keywords`, `cancel_after_ms` (stop the reply after that
 * long), `need_bytes` (the memory the check asks for, from this row on). Results go to
 * `runs/<plan name>.jsonl` in the external files dir (`plan/b.json` -> `runs/b.jsonl`), then
 * `AUTORUN_DONE` is logged.
 */
class AutoRun(private val vm: VoiceViewModel, private val root: File) {
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
          vm.ui.first {
            it.engine is EngineState.Ready || it.engine is EngineState.Failed
          }
          vm.setTask(Task.valueOf(row.optString("task", "TRANSCRIPT")))
          vm.setKeywords(row.optString("keywords", ""))
          vm.dismissMessage()
          val wav = File(root, row.getString("wav"))
          record.put("wav", row.getString("wav"))
          val samples =
            runCatching { Wav.decode(wav.readBytes()) }
              .getOrElse { e ->
                record.put("error", "${e.javaClass.simpleName}: ${e.message}")
                out.appendText(record.toString() + "\n")
                Log.i(TAG, "AUTORUN_ROW ${record.optString("tag")} error=${e.message}")
                null
              } ?: continue
          record.put("engine", vm.ui.value.engine.toString())
          val job =
            vm.afterCapture(
              samples,
              VoiceViewModel.peak(samples),
              "file",
              note = null,
              record = record,
              cancelAfterMs = if (row.has("cancel_after_ms")) row.getLong("cancel_after_ms") else null,
            )
          job?.join()
          record.put("message", vm.ui.value.message ?: JSONObject.NULL)
          out.appendText(record.toString() + "\n")
          Log.i(
            TAG,
            "AUTORUN_ROW ${record.optString("tag")} total_ms=${record.opt("total_ms")} " +
              "refused=${record.has("refused")} error=${record.opt("error")}",
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
    const val TAG = VoiceViewModel.TAG
  }
}
