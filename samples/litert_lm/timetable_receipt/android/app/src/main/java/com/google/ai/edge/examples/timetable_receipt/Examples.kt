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

import android.content.Context
import org.json.JSONObject

/** One bundled example: a picture in the assets and the question that goes with it. */
data class Example(
  val id: String,
  val title: String,
  val file: String,
  val task: Task,
  val after: String,
  val people: Int,
)

object Examples {
  fun load(context: Context): List<Example> {
    val json = context.assets.open("examples.json").bufferedReader().use { it.readText() }
    val list = JSONObject(json).getJSONArray("examples")
    return (0 until list.length()).map { i ->
      val o = list.getJSONObject(i)
      Example(
        id = o.getString("id"),
        title = o.getString("title"),
        file = "examples/" + o.getString("file"),
        task = Task.parse(o.getString("task")) ?: Task.ASK_ANYTHING,
        after = o.optString("after", "09:00"),
        people = o.optInt("people", 2),
      )
    }
  }
}
