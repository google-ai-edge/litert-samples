/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.asr

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class FileUtilTest {

  @Rule @JvmField val tempFolder = TemporaryFolder()

  @Test
  fun testReadOrDownloadFile_localExists() {
    val tempDir = tempFolder.newFolder("files")
    val localFile = File(tempDir, "test.txt")
    localFile.writeText("local content")

    val result =
      readOrDownloadFileInternal(
        filesDir = tempDir,
        assetsLoader = { null },
        localPath = "test.txt",
        remoteUrl = "",
      )
    assertEquals("local content", result)
  }

  @Test
  fun testReadOrDownloadFile_assetsFallback() {
    val tempDir = tempFolder.newFolder("files")

    val result =
      readOrDownloadFileInternal(
        filesDir = tempDir,
        assetsLoader = { "asset content".byteInputStream() },
        localPath = "test.txt",
        remoteUrl = "",
      )
    assertEquals("asset content", result)
  }

  @Test(expected = java.io.FileNotFoundException::class)
  fun testReadOrDownloadFile_fileNotFound() {
    val tempDir = tempFolder.newFolder("files")

    readOrDownloadFileInternal(
      filesDir = tempDir,
      assetsLoader = { null },
      localPath = "non_existent.txt",
      remoteUrl = "",
    )
  }

  @Test
  fun testOpenOrDownloadFile_localExists() {
    val tempDir = tempFolder.newFolder("files")
    val localFile = File(tempDir, "test.txt")
    localFile.writeText("local content")

    val stream =
      openOrDownloadFileInternal(
        filesDir = tempDir,
        assetsLoader = { null },
        localPath = "test.txt",
        remoteUrl = "",
      )
    val result = stream.use { input ->
      java.io.BufferedReader(java.io.InputStreamReader(input)).readText()
    }
    assertEquals("local content", result)
  }

  @Test
  fun testOpenOrDownloadFile_assetsFallback() {
    val tempDir = tempFolder.newFolder("files")

    val stream =
      openOrDownloadFileInternal(
        filesDir = tempDir,
        assetsLoader = { "asset content".byteInputStream() },
        localPath = "test.txt",
        remoteUrl = "",
      )
    val result = stream.use { input ->
      java.io.BufferedReader(java.io.InputStreamReader(input)).readText()
    }
    assertEquals("asset content", result)
  }
}
