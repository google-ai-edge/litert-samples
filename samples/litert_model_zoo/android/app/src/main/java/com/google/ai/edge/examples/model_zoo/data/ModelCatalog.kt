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

import java.net.URI
import org.json.JSONObject

data class ModelFile(val name: String, val url: String, val bytes: Long, val sha256: String?)

data class ModelLicense(val name: String, val url: String)

data class ComponentLicense(val component: String, val name: String, val url: String)

data class ModelEntry(
  val task: String,
  val taskId: String,
  val group: String,
  val model: String,
  val files: List<ModelFile>,
  val license: ModelLicense,
  val upstream: String,
  val modelCard: String,
  val backend: String,
  val inputKind: String,
  val outputKind: String,
  val componentLicenses: List<ComponentLicense> = emptyList(),
) {
  val totalBytes: Long
    get() = files.sumOf { it.bytes }

  val canDownload: Boolean
    get() = files.isNotEmpty() && files.all { it.sha256 != null && it.bytes > 0 }
}

data class ModelCatalog(
  val schemaVersion: Int,
  val runtimeVersion: String,
  val tasks: List<ModelEntry>,
) {
  companion object {
    private val safeName = Regex("[A-Za-z0-9][A-Za-z0-9_.-]*")
    private val safeId = Regex("[a-z0-9]+(?:-[a-z0-9]+)*")
    private val hash = Regex("[a-f0-9]{64}")
    private val allowedLicenses =
      setOf(
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "BSD-3-Clause-Clear",
        "Clear-BSD",
      )

    fun parse(json: String): ModelCatalog {
      val root = JSONObject(json)
      require(root.getInt("schemaVersion") == 1) { "Unsupported catalog schema" }
      val tasks = root.getJSONArray("tasks")
      val entries =
        (0 until tasks.length()).map { index ->
          val obj = tasks.getJSONObject(index)
          val array = obj.getJSONArray("files")
          val files =
            (0 until array.length()).map { fileIndex ->
              val file = array.getJSONObject(fileIndex)
              ModelFile(
                file.getString("name"),
                file.getString("url"),
                file.getLong("bytes"),
                if (file.isNull("sha256")) null else file.getString("sha256"),
              )
            }
          val license =
            requireNotNull(obj.optJSONObject("license")) { "Every model needs a license" }.let {
              ModelLicense(it.getString("name"), it.getString("url"))
            }
          ModelEntry(
            obj.getString("task"),
            obj.getString("taskId"),
            obj.getString("group"),
            obj.getString("model"),
            files,
            license,
            obj.optString("upstream", ""),
            obj.optString("modelCard", ""),
            obj.getString("backend"),
            obj.getString("inputKind"),
            obj.getString("outputKind"),
            obj.optJSONArray("componentLicenses")?.let { licenses ->
              (0 until licenses.length()).map { componentIndex ->
                val component = licenses.getJSONObject(componentIndex)
                ComponentLicense(
                  component.getString("component"),
                  component.getString("name"),
                  component.getString("url"),
                )
              }
            } ?: emptyList(),
          )
        }
      return ModelCatalog(root.getInt("schemaVersion"), root.getString("runtimeVersion"), entries)
        .also { it.validate() }
    }
  }

  fun validate() {
    require(tasks.map { it.taskId }.distinct().size == tasks.size) { "Duplicate task ID" }
    require(runtimeVersion == "2.2.0") { "Catalog and app runtime must match" }
    tasks.forEach { entry ->
      require(safeId.matches(entry.taskId)) { "Unsafe task ID" }
      require(entry.group in setOf("Vision", "Audio")) { "Unknown task group" }
      require(entry.backend in setOf("gpu", "cpu", "mixed")) { "Unknown backend" }
      require(entry.task.isNotBlank()) { "Missing task description" }
      require(entry.files.map { it.name }.distinct().size == entry.files.size) {
        "Duplicate model filename"
      }
      require(entry.license.name in allowedLicenses) {
        "Unapproved license: ${entry.license.name}"
      }
      require(URI(entry.license.url).scheme == "https") { "License URL must use HTTPS" }
      entry.componentLicenses.forEach { component ->
        require(component.name in allowedLicenses && URI(component.url).scheme == "https") {
          "Unapproved component license"
        }
      }
      entry.files.forEach { file ->
        val url = URI(file.url)
        require(
          safeName.matches(file.name) &&
            file.name != "." &&
            file.name != ".." &&
            !file.name.endsWith(".part")
        ) {
          "Unsafe model filename"
        }
        require(
          url.scheme == "https" &&
            url.host == "huggingface.co" &&
            url.userInfo == null &&
            url.port == -1
        ) {
          "Unexpected model host"
        }
        require(url.path.matches(Regex("/[^/]+/[^/]+/resolve/[A-Za-z0-9._-]+/.+"))) {
          "Model URL must identify an exact file at a pinned revision"
        }
        require(!url.path.endsWith(".onnx") && !url.path.endsWith(".litertlm")) {
          "Unsupported model runtime"
        }
        require(file.bytes > 0) { "Missing verified byte size" }
        require(file.sha256 == null || hash.matches(file.sha256)) { "Invalid SHA-256" }
      }
      require(entry.canDownload) { "Every model file needs its byte size and SHA-256" }
    }
  }
}
