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

package com.google.ai.edge.examples.timetable_receipt

/** The three questions the screen offers. */
enum class Task(val label: String) {
  NEXT_DEPARTURE("Next departure"),
  SPLIT_BILL("Split the bill"),
  ASK_ANYTHING("Question");

  /** The text sent with the picture. For the first two, the model only lists what it reads. */
  fun instruction(question: String): String = when (this) {
    NEXT_DEPARTURE ->
      "List every departure time in this timetable from top to bottom, one per line, in " +
        "24-hour HH:MM form. If there is no timetable in the picture, answer NONE."
    SPLIT_BILL ->
      "List each item on this receipt with its price, one per line as 'item: price'. Then " +
        "write the subtotal, tax and total lines the same way. If there is no receipt in the " +
        "picture, answer NONE."
    ASK_ANYTHING -> question.trim()
  }

  companion object {
    fun parse(name: String?): Task? = entries.firstOrNull { it.name.equals(name, ignoreCase = true) }
  }
}
