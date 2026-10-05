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
import org.json.JSONObject

data class ModelFile(val name: String, val url: String, val bytes: Long, val sha256: String)

data class ModelLicense(val name: String, val url: String)

data class ComponentLicense(val component: String, val name: String, val url: String)

/** One model of the loop and the files it downloads into filesDir/models/<id>. */
data class ModelEntry(
  val id: String,
  val role: String,
  val model: String,
  val runtime: String,
  val backend: String,
  val files: List<ModelFile>,
  val license: ModelLicense,
  val modelCard: String,
  val componentLicenses: List<ComponentLicense> = emptyList(),
) {
  val totalBytes: Long
    get() = files.sumOf { it.bytes }

  val canDownload: Boolean
    get() = files.isNotEmpty() && files.all { it.bytes > 0 }

  fun file(name: String): ModelFile =
    requireNotNull(files.firstOrNull { it.name == name }) { "$id has no file $name" }
}

/**
 * `assets/models.json`: the three models of the loop, every file with its exact URL, byte size and
 * SHA-256. The model files never ship in the APK.
 */
data class ModelCatalog(val schemaVersion: Int, val models: List<ModelEntry>) {
  fun entry(id: String): ModelEntry =
    requireNotNull(models.firstOrNull { it.id == id }) { "The catalog has no model $id" }

  companion object {
    const val ZIPFORMER = "zipformer"
    const val KITTEN = "kitten"
    const val GEMMA = "gemma"

    private val safeName = Regex("[A-Za-z0-9][A-Za-z0-9_.-]*")
    private val safeId = Regex("[a-z0-9]+(?:-[a-z0-9]+)*")
    private val hash = Regex("[a-f0-9]{64}")
    private val allowedLicenses = setOf("Apache-2.0", "MIT", "BSD-3-Clause-Clear")

    fun parse(json: String): ModelCatalog {
      val root = JSONObject(json)
      require(root.getInt("schemaVersion") == 1) { "Unsupported catalog schema" }
      val array = root.getJSONArray("models")
      val models =
        (0 until array.length()).map { index ->
          val obj = array.getJSONObject(index)
          val list = obj.getJSONArray("files")
          val files =
            (0 until list.length()).map { fileIndex ->
              val file = list.getJSONObject(fileIndex)
              ModelFile(
                file.getString("name"),
                file.getString("url"),
                file.getLong("bytes"),
                file.getString("sha256"),
              )
            }
          val license =
            requireNotNull(obj.optJSONObject("license")) { "Every model needs a license" }.let {
              ModelLicense(it.getString("name"), it.getString("url"))
            }
          val components =
            obj.optJSONArray("componentLicenses")?.let { licenses ->
              (0 until licenses.length()).map { componentIndex ->
                val component = licenses.getJSONObject(componentIndex)
                ComponentLicense(
                  component.getString("component"),
                  component.getString("name"),
                  component.getString("url"),
                )
              }
            } ?: emptyList()
          ModelEntry(
            obj.getString("id"),
            obj.getString("role"),
            obj.getString("model"),
            obj.getString("runtime"),
            obj.getString("backend"),
            files,
            license,
            obj.getString("modelCard"),
            components,
          )
        }
      return ModelCatalog(root.getInt("schemaVersion"), models).also { it.validate() }
    }
  }

  fun validate() {
    require(models.map { it.id }.distinct().size == models.size) { "Duplicate model ID" }
    models.forEach { entry ->
      require(safeId.matches(entry.id)) { "Unsafe model ID" }
      require(entry.runtime in setOf("litert", "litert_lm")) { "Unknown runtime" }
      require(entry.backend in setOf("gpu", "cpu")) { "Unknown backend" }
      require(entry.files.map { it.name }.distinct().size == entry.files.size) {
        "Duplicate model filename"
      }
      require(entry.license.name in allowedLicenses) {
        "Unapproved license: ${entry.license.name}"
      }
      require(URI(entry.license.url).scheme == "https") { "License URL must use HTTPS" }
      require(URI(entry.modelCard).scheme == "https") { "Model card URL must use HTTPS" }
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
          "Model URL must identify an exact file at a revision"
        }
        require(file.bytes > 0) { "Missing verified byte size" }
        require(hash.matches(file.sha256)) { "Invalid SHA-256" }
      }
      require(entry.canDownload) { "Every model file needs its byte size" }
    }
  }
}
