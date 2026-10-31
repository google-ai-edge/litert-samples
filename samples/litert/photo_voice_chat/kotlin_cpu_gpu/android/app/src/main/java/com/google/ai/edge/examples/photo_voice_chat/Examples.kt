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

import android.content.Context
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * The example photos in the app's assets (`examples/<photo>`, 512 x 512 PNG, listed in `examples.json`), each with three
 * spoken questions (`examples/<wav>`, 16 kHz mono, English, spoken by Kokoro-82M; README.md). A question's [text] is
 * what the recording says, shown on the screen as the question; the model hears only the recording.
 */
class ExamplePhoto(val id: String, val label: String, val photo: String, val questions: List<ExampleQuestion>) {
  fun load(context: Context): PhotoInput = Photos.fromAsset(context, "examples/$photo", label)

  companion object {
    fun list(context: Context): List<ExamplePhoto> {
      val items = JSONObject(context.assets.open("examples.json").bufferedReader().readText()).getJSONArray("photos")
      return List(items.length()) { i ->
        val o = items.getJSONObject(i)
        ExamplePhoto(o.getString("id"), o.getString("label"), o.getString("photo"), ExampleQuestion.list(o.getJSONArray("questions")))
      }
    }
  }
}

/**
 * The example videos in the app's assets (`examples/<video>`, listed under "videos" in `examples.json`; README.md), each
 * with a typed question (no recording: the question goes to the model as text).
 */
class ExampleVideo(val id: String, val label: String, val video: String, val bytes: Long, val questions: List<ExampleQuestion>) {
  /** The video as a file in the app's cache (the player and the reader open files), copied from the assets once. */
  fun file(context: Context): File {
    val f = File(context.cacheDir, "examples/$video")
    if (f.isFile && f.length() == bytes) return f
    f.parentFile?.mkdirs()
    val part = File(f.path + ".partial")
    context.assets.open("examples/$video").use { input -> part.outputStream().use { input.copyTo(it) } }
    check(part.length() == bytes) { "examples/$video: ${part.length()} bytes copied, $bytes expected" }
    check(part.renameTo(f)) { "could not write ${f.path}" }
    return f
  }

  companion object {
    fun list(context: Context): List<ExampleVideo> {
      val items = JSONObject(context.assets.open("examples.json").bufferedReader().readText()).optJSONArray("videos") ?: return emptyList()
      return List(items.length()) { i ->
        val o = items.getJSONObject(i)
        ExampleVideo(o.getString("id"), o.getString("label"), o.getString("video"), o.getLong("bytes"),
          ExampleQuestion.list(o.getJSONArray("questions")))
      }
    }
  }
}

/** An example question: [text] is what it asks; [wav] = its recording (null: it is asked as typed). */
class ExampleQuestion(val id: String, val text: String, val wav: String?) {
  fun load(context: Context): Clip =
    AudioIO.check("the example question", context.assets.open("examples/${wav ?: error("$id has no recording")}").use { AudioIO.readWav16k(it.readBytes()) })

  companion object {
    fun list(qs: JSONArray): List<ExampleQuestion> = List(qs.length()) { j ->
      val q = qs.getJSONObject(j)
      ExampleQuestion(q.getString("id"), q.getString("text"), q.optString("wav").ifEmpty { null })
    }
  }
}
