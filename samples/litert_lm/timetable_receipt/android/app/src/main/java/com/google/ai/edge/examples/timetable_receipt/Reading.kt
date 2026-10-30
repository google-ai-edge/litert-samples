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

import java.math.BigDecimal
import java.math.RoundingMode
import java.util.Locale

/** What the model read from a timetable, and the departure the app picked from it. */
data class TimetableReading(val times: List<String>, val next: String?, val saidNone: Boolean)

/** One line of a receipt as the model wrote it: a name and the line's amount. */
data class ReceiptLine(val name: String, val amount: BigDecimal, val role: Role) {
  enum class Role { ITEM, SUBTOTAL, TAX, TOTAL }
}

/** What the model read from a receipt, and the split the app computed from it. */
data class ReceiptReading(
  val lines: List<ReceiptLine>,
  val total: BigDecimal?,
  /** Where the total comes from: "the TOTAL line", "subtotal + tax" or "the items". */
  val totalSource: String,
  val people: Int,
  val perPerson: BigDecimal?,
  val itemsSum: BigDecimal,
  val subtotal: BigDecimal?,
  val tax: BigDecimal?,
  val saidNone: Boolean,
)

/**
 * The model only reads; the arithmetic is done here. The model lists the values it sees (one
 * departure time or one priced line per line of text), and these functions turn that text into
 * the values shown on the screen and the answer computed from them.
 */
object Reading {
  // No "[:" anywhere: Android's regex engine (ICU) reads "[:" as the start of a POSIX class.
  private val TIME = Regex("""(?<![\d.:])([01]?\d|2[0-3])(?:\.|:)([0-5]\d)(?![\d:])""")
  private val MONEY = Regex("""(\d{1,4}(?:,\d{3})*\.\d{2})(?!\d)""")
  private val NONE = Regex("""\bNONE\b""")
  private val LEADING_MARK = Regex("""^\s*(?:-|\*|\u2022|\d+\.|\d+\))\s*""")
  private val SUBTOTAL = Regex("""sub\s*-?\s*total""", RegexOption.IGNORE_CASE)
  private val TAX = Regex("""\btax\b""", RegexOption.IGNORE_CASE)
  private val TOTAL = Regex("""\btotal\b""", RegexOption.IGNORE_CASE)

  /** Every time in [text], in order, as HH:MM. */
  fun times(text: String): List<String> =
    TIME.findAll(text).map { m ->
      String.format(Locale.US, "%02d:%s", m.groupValues[1].toInt(), m.groupValues[2])
    }.toList()

  /** The departures read from [text] and the first one later than [after] (HH:MM). */
  fun timetable(text: String, after: String): TimetableReading {
    val times = times(text)
    val next = times.filter { it > after }.minOrNull()
    return TimetableReading(times, next, NONE.containsMatchIn(text) && times.isEmpty())
  }

  /**
   * The priced lines read from [text] (one value per line: the line's last amount with two
   * decimals) and the total split between [people], rounded to the cent (half up).
   */
  fun receipt(text: String, people: Int): ReceiptReading {
    val lines = mutableListOf<ReceiptLine>()
    for (raw in text.lines()) {
      val match = MONEY.findAll(raw).lastOrNull() ?: continue
      val amount = BigDecimal(match.value.replace(",", "")).setScale(2)
      val name = raw.substring(0, match.range.first)
        .replace(LEADING_MARK, "")
        .trim()
        .trimEnd(':', '-', '=', '$', ' ', '\t')
        .trim()
      val role = when {
        SUBTOTAL.containsMatchIn(name) -> ReceiptLine.Role.SUBTOTAL
        TAX.containsMatchIn(name) -> ReceiptLine.Role.TAX
        TOTAL.containsMatchIn(name) -> ReceiptLine.Role.TOTAL
        else -> ReceiptLine.Role.ITEM
      }
      lines += ReceiptLine(name, amount, role)
    }
    val items = lines.filter { it.role == ReceiptLine.Role.ITEM }
    val itemsSum = items.fold(BigDecimal.ZERO.setScale(2)) { acc, l -> acc + l.amount }
    val subtotal = lines.lastOrNull { it.role == ReceiptLine.Role.SUBTOTAL }?.amount
    val tax = lines.lastOrNull { it.role == ReceiptLine.Role.TAX }?.amount
    val totalLine = lines.lastOrNull { it.role == ReceiptLine.Role.TOTAL }?.amount
    val (total, source) = when {
      totalLine != null -> totalLine to "the TOTAL line"
      subtotal != null && tax != null -> (subtotal + tax) to "subtotal + tax"
      items.isNotEmpty() -> itemsSum to "the items"
      else -> null to ""
    }
    val n = people.coerceAtLeast(1)
    val per = total?.divide(BigDecimal(n), 2, RoundingMode.HALF_UP)
    return ReceiptReading(
      lines, total, source, n, per, itemsSum, subtotal, tax,
      NONE.containsMatchIn(text) && lines.isEmpty(),
    )
  }

  /** "HH:MM" (or "H:MM", "H.MM") typed by the user, as HH:MM; null when it is not a time. */
  fun normalizeTime(typed: String): String? {
    val m = Regex("""^\s*([01]?\d|2[0-3])(?:\.|:)([0-5]\d)\s*$""").find(typed) ?: return null
    return String.format(Locale.US, "%02d:%s", m.groupValues[1].toInt(), m.groupValues[2])
  }
}
