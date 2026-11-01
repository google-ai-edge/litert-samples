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

package com.google.ai.edge.examples.where_to_put

import android.content.Context
import org.json.JSONObject

/**
 * One suggested request for an example photo. [spotMask] (in the assets) is the region where the item can go: a
 * 1000 x 1000 picture over 0-1000 of the photo, white where the point counts as right. Only the measuring tool reads it.
 * [itemWidth] is how wide the item looks in this photo, in thousandths of the photo's width (set by hand from a like
 * object in the photo), for its drawing; null leaves the drawing's own width.
 */
data class Request(val id: String, val text: String, val spotMask: String, val itemWidth: Int?)

/** One bundled example: a photo in the assets, its credit line, and the requests that go with it. */
data class Example(val id: String, val title: String, val file: String, val credit: String, val requests: List<Request>)

object Examples {
  fun load(context: Context): List<Example> {
    val json = context.assets.open("examples.json").bufferedReader().use { it.readText() }
    val list = JSONObject(json).getJSONArray("examples")
    return (0 until list.length()).map { i ->
      val o = list.getJSONObject(i)
      val rs = o.getJSONArray("requests")
      Example(
        id = o.getString("id"),
        title = o.getString("title"),
        file = "examples/" + o.getString("file"),
        credit = o.getString("credit"),
        requests = (0 until rs.length()).map { j ->
          val r = rs.getJSONObject(j)
          Request(
            r.getString("id"), r.getString("text"), "examples/" + r.getString("spot_mask"),
            r.optInt("item_width", 0).takeIf { it > 0 },
          )
        },
      )
    }
  }

  /** The bundled request with this id, and its example. */
  fun request(examples: List<Example>, id: String): Pair<Example, Request>? {
    for (e in examples) for (r in e.requests) if (r.id == id) return Pair(e, r)
    return null
  }
}
