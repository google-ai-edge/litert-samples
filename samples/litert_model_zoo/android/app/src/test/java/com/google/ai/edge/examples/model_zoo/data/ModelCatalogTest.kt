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

import com.google.ai.edge.examples.model_zoo.audio.BatchAudioTasks
import com.google.ai.edge.examples.model_zoo.image.SingleImageTasks
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {
  @Test
  fun parsesDownloadableTaskWithoutLosingMetadata() {
    val catalog = ModelCatalog.parse(CatalogFixture.json().toString())
    assertEquals(1, catalog.schemaVersion)
    assertEquals("2.2.0", catalog.runtimeVersion)
    val entry = catalog.tasks.single()
    assertEquals("Example task", entry.task)
    assertEquals("example-task", entry.taskId)
    assertEquals("Vision", entry.group)
    assertEquals("Example model", entry.model)
    assertEquals("MIT", entry.license.name)
    assertEquals("gpu", entry.backend)
    assertEquals("image", entry.inputKind)
    assertEquals("boxes", entry.outputKind)
    assertEquals(3L, entry.totalBytes)
    assertTrue(entry.canDownload)
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsUnsupportedSchema() {
    parseModified { it.put("schemaVersion", 2) }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsRuntimeVersionConflict() {
    parseModified { it.put("runtimeVersion", "2.1.3") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsDuplicateTaskIds() {
    parseModified { it.getJSONArray("tasks").put(CatalogFixture.task(it)) }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsTaskPathTraversal() {
    parseModified { CatalogFixture.task(it).put("taskId", "../outside") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsModelFilenamePathTraversal() {
    parseModified { CatalogFixture.file(it).put("name", "../outside.tflite") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsPartialFilenameCollision() {
    parseModified {
      val files = CatalogFixture.task(it).getJSONArray("files")
      files.put(JSONObject(CatalogFixture.file(it).toString()).put("name", "model.tflite.part"))
    }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsNonHuggingFaceDownload() {
    parseModified { CatalogFixture.file(it).put("url", "https://example.com/model.tflite") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsUnencryptedDownload() {
    parseModified {
      CatalogFixture.file(it)
        .put("url", "http://huggingface.co/owner/model/resolve/main/model.tflite")
    }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsRepositoryPageInsteadOfExactFile() {
    parseModified { CatalogFixture.file(it).put("url", "https://huggingface.co/owner/model") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsAgplModel() {
    parseModified { CatalogFixture.task(it).getJSONObject("license").put("name", "AGPL-3.0") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsNonLiteRtModelFile() {
    parseModified {
      CatalogFixture.file(it)
        .put("url", "https://huggingface.co/owner/model/resolve/main/model.onnx")
        .put("name", "model.onnx")
    }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsModelWithMissingDigest() {
    parseModified { CatalogFixture.file(it).put("sha256", JSONObject.NULL) }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsMalformedDigest() {
    parseModified { CatalogFixture.file(it).put("sha256", "not-a-sha256") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsModelWithoutLicense() {
    parseModified { CatalogFixture.task(it).put("license", JSONObject.NULL) }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsModelWithoutFiles() {
    parseModified { CatalogFixture.task(it).put("files", org.json.JSONArray()) }
  }

  @Test
  fun packagedCatalogIsDownloadableAndPinned() {
    val catalog = packagedCatalog()
    assertTrue(catalog.tasks.isNotEmpty())
    assertTrue(catalog.tasks.all { it.canDownload })
    assertTrue(catalog.tasks.all { it.group in setOf("Vision", "Audio") })
    val pinned = Regex("https://huggingface.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+")
    assertTrue(catalog.tasks.all { task -> task.files.all { pinned.matches(it.url) } })
    catalog.tasks.forEach {
      assertTrue(it.license.name.isNotBlank() && it.license.url.startsWith("https://"))
      assertTrue(it.upstream.startsWith("https://"))
      assertTrue(it.modelCard.startsWith("https://"))
    }
    val panns = catalog.tasks.single { it.taskId == "audio-classification" }
    assertEquals("CC-BY-4.0", panns.license.name)
    assertTrue(OpenSourceCredits.pannsAttribution.contains("Qiuqiang Kong"))
    assertEquals("https://creativecommons.org/licenses/by/4.0/", OpenSourceCredits.ccByLicenseUrl)
    val video = catalog.tasks.single { it.taskId == "video-action-recognition" }
    assertEquals("cpu", video.backend)
    assertEquals("mixed", catalog.tasks.single { it.taskId == "image-tagging" }.backend)
    val asr = catalog.tasks.single { it.taskId == "speech-recognition" }
    assertEquals("Zipformer-small (CR-CTC)", asr.model)
    assertEquals(
      setOf("zipformer_ctc_small_fp16.tflite", "tokens.txt"),
      asr.files.map { it.name }.toSet(),
    )
  }

  @Test
  fun everyPackagedTaskHasAnEngineAndEveryEngineHasATask() {
    val ids = packagedCatalog().tasks.map { it.taskId }.toSet()
    val engines =
      SingleImageTasks.ids +
        BatchAudioTasks.ids +
        setOf("object-detection", "text-to-speech", "speech-recognition")
    assertEquals(emptySet<String>(), ids - engines)
    assertEquals(emptySet<String>(), engines - ids)
  }

  @Test
  fun mixedGraphBackendsArePreservedAndUnknownBackendsRejected() {
    val json = CatalogFixture.json()
    CatalogFixture.task(json).put("backend", "mixed")
    assertEquals("mixed", ModelCatalog.parse(json.toString()).tasks.single().backend)
    CatalogFixture.task(json).put("backend", "automatic")
    assertTrue(
      runCatching { ModelCatalog.parse(json.toString()) }.exceptionOrNull()
        is IllegalArgumentException
    )
  }

  private fun parseModified(modify: (JSONObject) -> Unit): ModelCatalog {
    val json = CatalogFixture.json()
    modify(json)
    return ModelCatalog.parse(json.toString())
  }

  private fun packagedCatalog(): ModelCatalog = ModelCatalog.parse(CatalogFixture.packagedJson())
}

internal object CatalogFixture {
  const val ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

  /** The catalog packaged in the app assets, on the test classpath. */
  fun packagedJson(): String =
    checkNotNull(CatalogFixture::class.java.getResourceAsStream("/models.json")) {
        "models.json must be on the test classpath"
      }
      .bufferedReader()
      .use { it.readText() }

  fun json(): JSONObject =
    JSONObject(
      """
    {
      "schemaVersion":1,
      "runtimeVersion":"2.2.0",
      "tasks":[{
        "task":"Example task",
        "taskId":"example-task",
        "group":"Vision",
        "model":"Example model",
        "files":[{
          "name":"model.tflite",
          "url":"https://huggingface.co/owner/model/resolve/main/model.tflite",
          "bytes":3,
          "sha256":"$ABC_SHA256"
        }],
        "license":{"name":"MIT","url":"https://example.com/LICENSE"},
        "upstream":"https://example.com/project",
        "modelCard":"https://huggingface.co/owner/model",
        "backend":"gpu",
        "inputKind":"image",
        "outputKind":"boxes"
      }]
    }
  """
        .trimIndent()
    )

  fun task(json: JSONObject): JSONObject = json.getJSONArray("tasks").getJSONObject(0)

  fun file(json: JSONObject): JSONObject = task(json).getJSONArray("files").getJSONObject(0)

  fun entry(): ModelEntry = ModelCatalog.parse(json().toString()).tasks.single()
}
