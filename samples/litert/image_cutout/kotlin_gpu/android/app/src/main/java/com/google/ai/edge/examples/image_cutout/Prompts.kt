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

// Port of WeVisOne-VL's build_prompt and SYSTEM_PROMPT (Tencent, Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

/**
 * The instructions the released checkpoint was trained with (`wevisone_vl.prompts.build_prompt` of
 * github.com/Tencent/WeVisOne-VL at 40f735b8) and the chat turn they go into.
 */
enum class Task(
  val label: String,
  val hint: String,
  val needsQuery: Boolean,
  /** The package's task name for this choice. */
  val packageTask: String,
) {
  CUT_OUT("Cut out", "What to cut out, e.g. the man on the left", true, "matting"),
  DETECT("Detect", "Labels, one per line or comma-separated, e.g. person, hat", true, "detection"),
  FIND("Find one", "A description, e.g. the yellow rectangle", true, "referring"),
  POINT("Point", "Labels, one per line or comma-separated, e.g. airplane, windsock", true, "point"),
  COUNT_BOXES("Count (boxes)", "What to count, e.g. blue circles", true, "count_box"),
  COUNT_POINTS("Count (points)", "What to count, e.g. people", true, "count_point"),
  READ_TEXT("Read text", "No words needed: every word in the photo is read", false, "ocr"),
  FIND_TEXT("Find text", "The text to find, e.g. EXIT 12", true, "ocr_grounding"),
}

/** The task of the package's task [name]; "segmentation" is another name for "matting". */
fun taskOf(name: String): Task {
  val key = name.trim().lowercase().let { if (it == "segmentation") "matting" else it }
  return Task.entries.firstOrNull { it.packageTask == key }
    ?: throw IllegalArgumentException("unknown task $name")
}

object Prompts {
  const val SYSTEM = "You are a helpful assistant."

  /**
   * Everything before the 256 image positions: the fixed system turn, then the user turn with the
   * image first.
   */
  const val BEFORE_IMAGE =
    "<|im_start|>system\n$SYSTEM<|im_end|>\n<|im_start|>user\n<|vision_start|>"

  /** Everything after the image: the instruction, then the assistant turn the model completes. */
  fun afterImage(instruction: String): String =
    "<|vision_end|>$instruction<|im_end|>\n<|im_start|>assistant\n"

  /**
   * The package's build_prompt(task, query) for [task]. Multi-label tasks take one label per line
   * (the package's own rule) or per comma (this app's input rule); the other tasks take the whole
   * text as one phrase.
   */
  fun instruction(task: Task, query: String): String {
    val text = query.trim()
    return when (task) {
      Task.DETECT ->
        "Locate all the instances that match the following description: " +
          labels(query).joinToString(". ") +
          "."
      Task.FIND -> "Locate the instances that match the following description: " + phrase(text)
      Task.POINT -> "point to: " + labels(query).joinToString(". ")
      Task.COUNT_BOXES ->
        "How many ${text.ifEmpty { "objects" }} in the image? Please count them with boxes."
      Task.COUNT_POINTS -> "How many ${text.ifEmpty { "objects" }} in the image?"
      Task.CUT_OUT ->
        "Locate the instances that match the following description and provide the alpha mattes: " +
          phrase(text)
      Task.READ_TEXT -> "detect all the word in the box format"
      Task.FIND_TEXT -> "Locate the following text in the image: $text"
    }
  }

  /** `_labels`: the non-empty stripped lines (and here also comma-separated parts). */
  fun labels(query: String): List<String> =
    query.split('\n', ',').map { it.trim() }.filter { it.isNotEmpty() }

  /** `_phrase`: the stripped text with a final "." unless it already ends in . ? or !. */
  fun phrase(text: String): String = if (text.isEmpty() || text.last() in ".?!") text else "$text."
}
