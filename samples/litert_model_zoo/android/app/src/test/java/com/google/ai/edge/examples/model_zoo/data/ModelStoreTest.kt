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

import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelStoreTest {
  @get:Rule val temporary = TemporaryFolder()

  @Test
  fun validatesByteCountAndKnownSha256Together() {
    val file = temporary.newFile("model.tflite").apply { writeText("abc") }
    val expected = CatalogFixture.entry().files.single()
    assertTrue(ModelStore.verify(file, expected))
    file.writeText("abd")
    assertFalse("Same length cannot bypass content integrity", ModelStore.verify(file, expected))
    file.writeText("ab")
    assertFalse("A short file cannot be ready", ModelStore.verify(file, expected))
  }

  @Test
  fun missingDigestNeverMarksPresentFileReady() = runBlocking {
    val entry =
      CatalogFixture.entry().let { original ->
        original.copy(files = original.files.map { it.copy(sha256 = null) })
      }
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val file = modelFile(store, entry).apply { writeText("abc") }
    assertFalse(ModelStore.verify(file, entry.files.single()))
    assertEquals(DownloadStatus.PAUSED, store.inspect(entry).status)
  }

  @Test
  fun inspectReportsPartialProgressWithoutDiscardingResumableBytes() = runBlocking {
    val entry = CatalogFixture.entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val part = File(modelFile(store, entry).path + ".part").apply { writeText("ab") }
    val state = store.inspect(entry)
    assertEquals(DownloadStatus.PAUSED, state.status)
    assertEquals(2L, state.receivedBytes)
    assertEquals(3L, state.totalBytes)
    assertArrayEquals("ab".toByteArray(), part.readBytes())
  }

  @Test
  fun promotesFullyDownloadedPartialFileAfterHashVerificationWithoutNetwork() = runBlocking {
    val entry = CatalogFixture.entry()
    val events = mutableListOf<String>()
    val store = ModelStore(temporary.newFolder(), eventLogger = { events.add(it) })
    val target = modelFile(store, entry)
    val part = File(target.path + ".part").apply { writeText("abc") }
    val states = mutableListOf<DownloadState>()
    store.download(entry) { states.add(it) }
    assertFalse(part.exists())
    assertArrayEquals("abc".toByteArray(), target.readBytes())
    assertEquals(listOf(DownloadStatus.VERIFYING, DownloadStatus.READY), states.map { it.status })
    assertEquals(DownloadStatus.READY, store.inspect(entry).status)
    val verified = org.json.JSONObject(events.last())
    assertEquals("sha_ok", verified.getString("event"))
    assertEquals(3L, verified.getLong("initialPartBytes"))
    assertEquals(entry.files.single().sha256, verified.getString("sha256"))
  }

  @Test
  fun corruptCompletePartialFileCannotBeCommitted() = runBlocking {
    val entry = CatalogFixture.entry()
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    val target = modelFile(store, entry)
    File(target.path + ".part").writeText("abd")
    val failed = runCatching { store.download(entry) {} }
    assertTrue(failed.exceptionOrNull() is IllegalStateException)
    assertFalse(target.exists())
    assertFalse(store.inspect(entry).status == DownloadStatus.READY)
  }

  @Test
  fun removingTaskLeavesOtherTaskFilesAndUpdatesStorageUsage() = runBlocking {
    val entry = CatalogFixture.entry()
    val other = entry.copy(taskId = "another-task")
    val store = ModelStore(temporary.newFolder(), eventLogger = {})
    modelFile(store, entry).writeText("abc")
    val retained = modelFile(store, other).apply { writeText("abc") }
    assertEquals(6L, store.storageBytes())
    store.delete(entry)
    assertFalse(store.directory(entry).exists())
    assertTrue(retained.exists())
    assertEquals(3L, store.storageBytes())
  }

  private fun modelFile(store: ModelStore, entry: ModelEntry): File {
    store.directory(entry).mkdirs()
    return File(store.directory(entry), entry.files.single().name)
  }
}
