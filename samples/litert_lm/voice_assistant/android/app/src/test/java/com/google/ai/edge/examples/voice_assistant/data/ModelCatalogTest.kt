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

package com.google.ai.edge.examples.voice_assistant.data

import java.net.URI
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {
  private fun packagedJson(): String =
    checkNotNull(ModelCatalogTest::class.java.getResourceAsStream("/models.json")) {
        "models.json must be on the test classpath"
      }
      .bufferedReader()
      .use { it.readText() }

  private fun packaged(): ModelCatalog = ModelCatalog.parse(packagedJson())

  @Test
  fun packagedCatalogHasTheThreeModelsOfTheLoop() {
    val catalog = packaged()
    assertEquals(1, catalog.schemaVersion)
    assertEquals(
      listOf(ModelCatalog.ZIPFORMER, ModelCatalog.KITTEN, ModelCatalog.GEMMA),
      catalog.models.map { it.id },
    )
    assertEquals("gpu", catalog.entry(ModelCatalog.ZIPFORMER).backend)
    assertEquals("cpu", catalog.entry(ModelCatalog.KITTEN).backend)
    assertEquals("litert_lm", catalog.entry(ModelCatalog.GEMMA).runtime)
    assertEquals(
      setOf("zipformer_ctc_fp16.tflite", "tokens.txt"),
      catalog.entry(ModelCatalog.ZIPFORMER).files.map { it.name }.toSet(),
    )
    assertEquals(7, catalog.entry(ModelCatalog.KITTEN).files.size)
    assertEquals(
      listOf("gemma-4-E2B-it.litertlm"),
      catalog.entry(ModelCatalog.GEMMA).files.map { it.name },
    )
    assertEquals(2588147712L, catalog.entry(ModelCatalog.GEMMA).totalBytes)
  }

  @Test
  fun everyFileHasItsSizeAndDigestAndAnHttpsHuggingFaceUrl() {
    val sha256 = Regex("[a-f0-9]{64}")
    packaged().models.forEach { entry ->
      entry.files.forEach { file ->
        assertTrue(file.name, file.bytes > 0)
        assertTrue(file.name, sha256.matches(file.sha256))
        val url = URI(file.url)
        assertEquals(file.name, "https", url.scheme)
        assertEquals(file.name, "huggingface.co", url.host)
        assertTrue(file.name, url.path.endsWith("/" + file.name))
      }
      assertTrue(entry.license.url.startsWith("https://"))
      assertTrue(entry.modelCard.startsWith("https://huggingface.co/"))
    }
  }

  @Test
  fun everyFileIsPinnedToACommit() {
    val pinned = Regex("https://huggingface.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+")
    packaged().models.flatMap { it.files }.forEach { assertTrue(it.url, pinned.matches(it.url)) }
    // The speaker's G2P files come from litert-community/Matcha-TTS.
    val g2p = packaged().entry(ModelCatalog.KITTEN).files.filter { "Matcha-TTS" in it.url }
    assertEquals(
      setOf("dp_g2p_matcha_fp16.tflite", "g2p_dict.txt.gz", "g2p_meta.json"),
      g2p.map { it.name }.toSet(),
    )
  }

  @Test
  fun theSymbolTableAssetIsKittensTable() {
    // Not on the Hub: the app ships it; its size and SHA-256 are the speaker descriptor's.
    val bytes =
      checkNotNull(ModelCatalogTest::class.java.getResourceAsStream("/symbols.json")) {
          "symbols.json must be on the test classpath"
        }
        .use { it.readBytes() }
    assertEquals(1613, bytes.size)
    val sha256 = MessageDigest.getInstance("SHA-256").digest(bytes)
    assertEquals(
      "366c66d45b1f4ec25463e08d5849921ce364baab2a8a9891df7f9cec602db2fa",
      sha256.joinToString("") { "%02x".format(it) },
    )
    assertEquals(178, JSONObject(String(bytes, Charsets.UTF_8)).getJSONArray("symbols").length())
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsAFilenameThatLeavesTheModelDirectory() {
    parseModified { file(it).put("name", "../outside.tflite") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsAnUnencryptedDownload() {
    parseModified { file(it).put("url", file(it).getString("url").replace("https:", "http:")) }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsADownloadFromAnotherHost() {
    parseModified { file(it).put("url", "https://example.com/zipformer_ctc_fp16.tflite") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsAMalformedDigest() {
    parseModified { file(it).put("sha256", "not-a-sha256") }
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsAnUnapprovedLicense() {
    parseModified {
      it.getJSONArray("models").getJSONObject(0).getJSONObject("license").put("name", "AGPL-3.0")
    }
  }

  private fun file(root: JSONObject): JSONObject =
    root.getJSONArray("models").getJSONObject(0).getJSONArray("files").getJSONObject(0)

  private fun parseModified(modify: (JSONObject) -> Unit): ModelCatalog {
    val json = JSONObject(packagedJson())
    modify(json)
    return ModelCatalog.parse(json.toString())
  }
}
