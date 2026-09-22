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

package com.google.ai.edge.examples.model_zoo.data

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class PackagedCatalogTest {
  @Test
  fun packagedCatalogUsesOnlyTheDocumentedFields() {
    val json = CatalogFixture.packagedJson()
    val root = JSONObject(json)
    assertEquals(setOf("schemaVersion", "runtimeVersion", "tasks"), root.keys().asSequence().toSet())
    val fields =
      setOf(
        "task",
        "taskId",
        "group",
        "model",
        "files",
        "license",
        "upstream",
        "modelCard",
        "backend",
        "inputKind",
        "outputKind",
        "componentLicenses",
      )
    val tasks = root.getJSONArray("tasks")
    for (i in 0 until tasks.length()) {
      assertTrue(fields.containsAll(tasks.getJSONObject(i).keys().asSequence().toSet()))
      val files = tasks.getJSONObject(i).getJSONArray("files")
      for (j in 0 until files.length()) {
        assertEquals(
          setOf("name", "url", "bytes", "sha256"),
          files.getJSONObject(j).keys().asSequence().toSet(),
        )
      }
    }
  }
}
