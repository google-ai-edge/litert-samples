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

// Vendored from https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT/blob/db80197282d11373df084c0ceed67a54544cfa84/android/app/src/main/java/com/gliner25decide/DecideSchema.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.gliner

/**
 * One classification head, mirroring gliner2 2.0.0 `Schema.classification(task, labels, ...)` as
 * `classify_text` builds it from a tasks dict: `{task: [labels]}` or `{task: {"labels": [...] |
 * {label: description}, "multi_label": bool, "cls_threshold": float, "prompt": str}}`.
 *
 * [clsThreshold] stays a Double: gliner2 compares each float32 probability, widened to a Python
 * float, against the Python float threshold. A Float threshold would round 0.7 down and accept a
 * probability of 0.69999999 that gliner2 rejects.
 */
data class Task(
  val name: String,
  val labels: List<String>,
  val multiLabel: Boolean = false,
  val clsThreshold: Double = 0.5,
  val prompt: String? = null,
  /** Label → description, in the order gliner2 appends them (the dict order). */
  val labelDescriptions: Map<String, String>? = null,
  /** gliner2 `class_act`; AUTO = sigmoid for multi-label, softmax otherwise. */
  val activation: Activation = Activation.AUTO,
)

/** gliner2 2.0.0 `class_act` values accepted by `_extract_classification_result`. */
enum class Activation(val key: String) {
  /** Sigmoid for multi-label heads, softmax otherwise. */
  AUTO("auto"),
  /** Always sigmoid. */
  SIGMOID("sigmoid"),
  /** Always softmax. */ SOFTMAX("softmax");

  companion object {
    /** The activation for a gliner2 `class_act` string. */
    fun fromKey(key: String): Activation =
      entries.firstOrNull { it.key == key }
        ?: throw IllegalArgumentException("Unknown class_act $key")
  }
}

/**
 * gliner2 2.0.0 `processor.py:SchemaTransformer._transform_schema` for classification heads at
 * inference (`example_mode = "both"`, no few-shot examples), plus the checks that keep the Kotlin
 * host from silently differing from gliner2 on inputs gliner2 itself mishandles.
 */
object DecideSchema {
  /** Marker before each task's prompt string; its position is dropped from label routing. */
  const val P_TOKEN = "[P]"

  /** Marker before each label; the graph reads the encoder output at these positions. */
  const val L_TOKEN = "[L]"

  /** Separator inside a prompt string before a label description. */
  const val DESC_TOKEN = "[DESCRIPTION]"

  /** Separator between task schemas. */
  const val SEP_STRUCT = "[SEP_STRUCT]"

  /** Separator between the schemas and the text words. */
  const val SEP_TEXT = "[SEP_TEXT]"

  /**
   * `prompt_str`: the task name, then `": " + prompt` when a prompt is set, then `" [DESCRIPTION]
   * label: description"` for every described label, in description order. gliner2 tokenizes this
   * whole string as ONE schema token.
   */
  fun promptString(task: Task): String {
    val builder = StringBuilder(task.name)
    if (!task.prompt.isNullOrEmpty()) {
      builder.append(": ").append(task.prompt)
    }
    task.labelDescriptions
      ?.takeIf { it.isNotEmpty() }
      ?.forEach { (label, description) ->
        if (label in task.labels) {
          builder
            .append(' ')
            .append(DESC_TOKEN)
            .append(' ')
            .append(label)
            .append(": ")
            .append(description)
        }
      }
    return builder.toString()
  }

  /** `( [P] prompt_str ( [L] label1 [L] label2 … ) )`, one entry per gliner2 schema token. */
  fun schemaTokens(task: Task): List<String> {
    val tokens = arrayListOf("(", P_TOKEN, promptString(task), "(")
    for (label in task.labels) {
      tokens.add(L_TOKEN)
      tokens.add(label)
    }
    tokens.add(")")
    tokens.add(")")
    return tokens
  }

  /**
   * Rejects task lists that gliner2 2.0.0 cannot classify or would classify under another task's
   * configuration: no tasks, duplicate names (a Python dict cannot hold them), a head without
   * labels (gliner2's argmax fails on an empty tensor), more labels than the graph's [labelSlots],
   * or a name that `_resolve_classification_config` maps to a different head.
   */
  fun validate(tasks: List<Task>, labelSlots: Int) {
    require(tasks.isNotEmpty()) { "Add at least one task." }
    val names = HashSet<String>()
    for (task in tasks) {
      require(names.add(task.name)) { "Task \"${task.name}\" appears twice." }
      require(task.labels.isNotEmpty()) { "Task \"${task.name}\" has no labels." }
    }
    val labelCount = tasks.sumOf { it.labels.size }
    require(labelCount <= labelSlots) {
      "$labelCount labels exceed the graph's $labelSlots label slots."
    }
    tasks.forEachIndexed { index, task ->
      val resolved = resolveConfigIndex(promptString(task), tasks)
      require(resolved == index) {
        "Task \"${task.name}\" would be decoded with the settings of \"${
          resolved?.let { tasks[it].name }
        }\" by gliner2 2.0.0; rename one of them."
      }
    }
  }

  /**
   * Port of gliner2 2.0.0 `inference/runtime.py:_resolve_classification_config`: the longest task
   * name that prefixes `prompt_str` and ends at its end, a colon or a space; otherwise the first
   * task name that prefixes it. Returns the task index, or null.
   */
  fun resolveConfigIndex(promptString: String, tasks: List<Task>): Int? {
    var best: Int? = null
    tasks.forEachIndexed { index, task ->
      if (task.name.isEmpty() || !promptString.startsWith(task.name)) {
        return@forEachIndexed
      }
      val rest = promptString.substring(task.name.length)
      if (rest.isEmpty() || rest[0] == ':' || rest[0] == ' ') {
        if (best == null || task.name.length > tasks[best!!].name.length) {
          best = index
        }
      }
    }
    return best ?: tasks.indices.firstOrNull { promptString.startsWith(tasks[it].name) }
  }

  /**
   * Parses the app's task editor: one task per line, `task: label1, label2, …`, followed by
   * optional ` | multi` or ` | multi 0.4` (multi-label with a threshold) and ` | prompt: …`
   * segments. Blank lines are skipped. Labels cannot contain commas in this format.
   */
  fun parseTaskLines(text: String): List<Task> =
    text
      .lines()
      .filter { it.isNotBlank() }
      .mapIndexed { lineIndex, line ->
        val segments = line.split(" | ")
        val head = segments.first()
        val colon = head.indexOf(':')
        require(colon > 0) { "Line ${lineIndex + 1}: write \"task: label1, label2\"." }
        val name = head.substring(0, colon).trim()
        require(name.isNotEmpty()) { "Line ${lineIndex + 1}: the task name is empty." }
        val labels =
          head.substring(colon + 1).split(',').map { it.trim() }.filter { it.isNotEmpty() }
        var multiLabel = false
        var threshold = 0.5
        var prompt: String? = null
        for (segment in segments.drop(1)) {
          val option = segment.trim()
          when {
            option == "multi" -> multiLabel = true
            option.startsWith("multi ") -> {
              multiLabel = true
              threshold =
                option.removePrefix("multi ").trim().toDoubleOrNull()?.takeIf { it in 0.0..1.0 }
                  ?: throw IllegalArgumentException(
                    "Line ${lineIndex + 1}: the multi threshold must be a number from 0 to 1."
                  )
            }
            option.startsWith("prompt:") -> prompt = option.removePrefix("prompt:").trim()
            else ->
              throw IllegalArgumentException("Line ${lineIndex + 1}: unknown option \"$option\".")
          }
        }
        Task(name, labels, multiLabel, threshold, prompt)
      }
}
