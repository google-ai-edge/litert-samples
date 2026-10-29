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

package com.google.ai.edge.examples.math_to_latex

import com.google.ai.edge.examples.math_to_latex.data.DownloadSafety
import com.google.ai.edge.examples.math_to_latex.data.ModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class CatalogAndMemoryTest {
  @Test
  fun catalogNamesOneFileAtARevision() {
    val json = javaClass.classLoader!!.getResource("models.json")!!.readText()
    val entry = ModelCatalog.parse(json).entry(ModelCatalog.OCR)
    assertEquals(1, entry.files.size)
    val file = entry.files.single()
    assertEquals("MiniCPM-OCR-1.0-VL_int8.litertlm", file.name)
    assertTrue(
      file.url.startsWith("https://huggingface.co/litert-community/MiniCPM-OCR-1.0/resolve/")
    )
    assertEquals(1_997_636_640L, file.bytes)
    assertEquals(64, file.sha256.length)
    assertEquals("Apache-2.0", entry.license.name)
  }

  @Test
  fun catalogRefusesAnotherHost() {
    val json =
      javaClass.classLoader!!
        .getResource("models.json")!!
        .readText()
        .replace("https://huggingface.co/", "https://example.com/")
    try {
      ModelCatalog.parse(json)
      fail("a file on another host was accepted")
    } catch (e: IllegalArgumentException) {
      assertTrue(e.message!!.contains("host") || e.message!!.contains("URL"))
    }
  }

  @Test
  fun downloadNeedsTwiceTheFile() {
    assertTrue(DownloadSafety.hasSpace(2_000_000_000L, 4_000_000_000L))
    assertFalse(DownloadSafety.hasSpace(2_000_000_000L, 3_999_999_999L))
  }

  @Test
  fun memoryCheckRefusesBelowTheNeed() {
    MemoryCheck(neededBytes = 3_000_000_000L, read = { 3_000_000_000L }).require()
    try {
      MemoryCheck(neededBytes = 3_000_000_000L, read = { 2_999_999_999L }).require()
      fail("2.99 GB passed a 3 GB check")
    } catch (e: NotEnoughMemoryException) {
      assertTrue(e.message!!.contains("Nothing was loaded"))
    }
  }

  @Test
  fun memoryCheckPassesWhenUnreadable() {
    MemoryCheck(neededBytes = Long.MAX_VALUE, read = { null }).require()
  }
}
