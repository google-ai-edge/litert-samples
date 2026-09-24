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

package com.google.ai.edge.examples.model_zoo

/** English result labels; task outputs and numerical processing do not depend on these strings. */
object ResultCounts {
  fun objects(count: Int): String = "${label(count, "object", "objects")} detected"

  fun tags(count: Int): String = label(count, "tag", "tags")

  fun boxes(count: Int): String = label(count, "box", "boxes")

  fun notes(count: Int): String = label(count, "note", "notes")

  fun matches(count: Int): String = label(count, "match", "matches")

  private fun label(count: Int, singular: String, plural: String): String {
    require(count >= 0)
    return "$count ${if (count == 1) singular else plural}"
  }
}
