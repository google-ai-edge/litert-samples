/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.google.ai.edge.examples.asr

import android.content.Context
import java.io.IOException

/** Open a file from the assets folder, or from app's private storage. */
fun existsInAssets(context: Context, path: String): Boolean =
  try {
    context.assets.open(path).close()
    true
  } catch (e: IOException) {
    false
  }

/** Reads a file from local storage, assets, or from a remote URL if not present. */
fun readOrDownloadFile(context: Context, localPath: String, remoteUrl: String) =
  readOrDownloadFileInternal(
    filesDir = context.filesDir,
    assetsLoader = { path ->
      if (existsInAssets(context, path)) context.assets.open(path) else null
    },
    localPath = localPath,
    remoteUrl = remoteUrl,
  )

/** Internal testable version of readOrDownloadFile. */
fun readOrDownloadFileInternal(
  filesDir: java.io.File,
  assetsLoader: (String) -> java.io.InputStream?,
  localPath: String,
  remoteUrl: String,
) =
  openOrDownloadFileInternal(filesDir, assetsLoader, localPath, remoteUrl).use { input ->
    java.io.BufferedReader(java.io.InputStreamReader(input)).readText()
  }

/** Downloads a file from a remote URL to a local file. */
fun openOrDownloadFile(context: Context, localPath: String, remoteUrl: String) =
  openOrDownloadFileInternal(
    filesDir = context.filesDir,
    assetsLoader = { path ->
      if (existsInAssets(context, path)) context.assets.open(path) else null
    },
    localPath = localPath,
    remoteUrl = remoteUrl,
  )

/** Internal testable version of openOrDownloadFile. */
fun openOrDownloadFileInternal(
  filesDir: java.io.File,
  assetsLoader: (String) -> java.io.InputStream?,
  localPath: String,
  remoteUrl: String,
): java.io.InputStream {
  val localFile = java.io.File(filesDir, localPath)
  if (localFile.exists()) {
    return localFile.inputStream()
  }

  val assetStream = assetsLoader(localPath)
  if (assetStream != null) {
    return assetStream
  }

  if (remoteUrl.isNotEmpty()) {
    val connection = java.net.URL(remoteUrl).openConnection()
    connection.connect()
    connection.getInputStream().use { input ->
      localFile.parentFile?.mkdirs()
      localFile.outputStream().use { output -> input.copyTo(output) }
    }
    return localFile.inputStream()
  }

  throw java.io.FileNotFoundException("Resource file not found: $localPath")
}
