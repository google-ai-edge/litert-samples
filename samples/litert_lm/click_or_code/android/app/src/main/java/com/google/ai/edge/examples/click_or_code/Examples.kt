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

import org.json.JSONObject

/** A bundled screenshot and the requests to try on it. */
data class Example(val id: String, val name: String, val file: String, val requests: List<String>)

/** `assets/examples/examples.json`. */
object Examples {
  fun parse(json: String): List<Example> {
    val array = JSONObject(json).getJSONArray("examples")
    return (0 until array.length()).map { i ->
      val e = array.getJSONObject(i)
      val r = e.getJSONArray("requests")
      Example(
        id = e.getString("id"),
        name = e.getString("name"),
        file = e.getString("file"),
        requests = (0 until r.length()).map { r.getString(it) },
      )
    }
  }
}
