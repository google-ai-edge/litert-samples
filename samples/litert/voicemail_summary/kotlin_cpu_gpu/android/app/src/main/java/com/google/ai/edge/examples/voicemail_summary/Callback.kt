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

package com.google.ai.edge.examples.voicemail_summary

/**
 * Reading an answer on the host: the callback number the screen offers to dial, and the checks the measuring tool
 * writes for the bundled voicemails (whose caller and number are known).
 */
object Callback {
  // a space, a dot, any hyphen or dash (the model may write U+2011, the non-breaking hyphen) or a no-break space
  private const val SEP = """[ .\-  ‐-―−]"""
  private const val NOT_BEFORE_AFTER = """[\d\-‐-―−]"""
  private val NUMBER =
    Regex("""(?<!$NOT_BEFORE_AFTER)(?:\+?1$SEP?)?(?:\(?\d{3}\)?$SEP?)?\d{3}$SEP?\d{4}(?!$NOT_BEFORE_AFTER)""")

  /** [text] with every phone number written with digits replaced by [spoken] of it. */
  fun replaceNumbers(text: String, spoken: (String) -> String): String = NUMBER.replace(text) { spoken(it.value) }

  /** The first phone number written with digits in [text] (7 digits, or 10 with an area code), as dialed; or null. */
  fun findNumber(text: String): String? {
    val m = NUMBER.find(text) ?: return null
    val d = m.value.filter { it.isDigit() }
    return when (d.length) {
      7 -> d.substring(0, 3) + "-" + d.substring(3)
      10 -> d.substring(0, 3) + "-" + d.substring(3, 6) + "-" + d.substring(6)
      11 -> d.substring(1, 4) + "-" + d.substring(4, 7) + "-" + d.substring(7)
      else -> null
    }
  }

  private val ONES = mapOf("zero" to 0, "oh" to 0, "o" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5,
    "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9)
  private val TEENS = mapOf("ten" to 10, "eleven" to 11, "twelve" to 12, "thirteen" to 13, "fourteen" to 14, "fifteen" to 15,
    "sixteen" to 16, "seventeen" to 17, "eighteen" to 18, "nineteen" to 19)
  private val TENS = mapOf("twenty" to 2, "thirty" to 3, "forty" to 4, "fifty" to 5, "sixty" to 6, "seventy" to 7, "eighty" to 8,
    "ninety" to 9)

  /** Lower case, everything but letters, digits and apostrophes as spaces, a possessive 's dropped. */
  fun words(text: String): List<String> = text.lowercase().replace(Regex("[^a-z0-9']+"), " ").trim().split(' ')
    .filter { it.isNotEmpty() }.map { it.removeSuffix("'s").trim('\'') }.filter { it.isNotEmpty() }

  /**
   * Every digit of [text] in order, spoken or written (the measuring tool's rule): digit strings as they are; zero / oh / o -> 0,
   * one..nine; ten..nineteen -> two digits; twenty..ninety -> its digit and the next one..nine (else 0); double / triple
   * X -> XX / XXX; X hundred [and] -> X then the tens and ones after it (else 00); every other word adds nothing.
   */
  fun digitStream(text: String): String {
    val w = words(text)
    val out = StringBuilder()
    var i = 0
    fun onesAt(j: Int): Int? = w.getOrNull(j)?.let { ONES[it] ?: it.takeIf { s -> s.length == 1 && s[0].isDigit() }?.toInt() }
    while (i < w.size) {
      val t = w[i]
      when {
        t.all { it.isDigit() } -> out.append(t)
        t in ONES -> out.append(ONES.getValue(t))
        t in TEENS -> out.append(TEENS.getValue(t))
        t in TENS -> {
          out.append(TENS.getValue(t))
          val o = w.getOrNull(i + 1)?.let { ONES[it] }
          if (o != null && o > 0) {
            out.append(o)
            i++
          } else out.append(0)
        }
        (t == "double" || t == "triple") && onesAt(i + 1) != null -> {
          repeat(if (t == "double") 2 else 3) { out.append(onesAt(i + 1)) }
          i++
        }
        t == "hundred" -> {
          var j = i + 1
          if (w.getOrNull(j) == "and") j++
          val n = w.getOrNull(j)
          when {
            n != null && n in TEENS -> { out.append(TEENS.getValue(n)); i = j }
            n != null && n in TENS -> {
              out.append(TENS.getValue(n))
              val o = w.getOrNull(j + 1)?.let { ONES[it] }
              if (o != null && o > 0) { out.append(o); i = j + 1 } else { out.append(0); i = j }
            }
            n != null && ONES[n]?.let { it > 0 } == true -> { out.append(0).append(ONES.getValue(n)); i = j }
            else -> out.append("00")
          }
        }
      }
      i++
    }
    return out.toString()
  }

  /** The number's digits (e.g. "555-0147" -> "5550147") appear in a row in [text]'s digit stream. */
  fun hasNumber(text: String, number: String): Boolean = digitStream(text).contains(number.filter { it.isDigit() })

  /** Both the first and the last name of [caller] appear as words in [text]. */
  fun namesCaller(text: String, caller: String): Boolean {
    val have = words(text).toSet()
    return words(caller).all { it in have }
  }

  // ---- answers in Spanish (and English): accents dropped, Spanish number words, "o" is not a digit ("or") ----------

  private val ES_ONES = mapOf("cero" to 0, "uno" to 1, "una" to 1, "un" to 1, "dos" to 2, "tres" to 3, "cuatro" to 4,
    "cinco" to 5, "seis" to 6, "siete" to 7, "ocho" to 8, "nueve" to 9)
  private val ES_TEENS = mapOf("diez" to 10, "once" to 11, "doce" to 12, "trece" to 13, "catorce" to 14, "quince" to 15,
    "dieciseis" to 16, "diecisiete" to 17, "dieciocho" to 18, "diecinueve" to 19, "veinte" to 20, "veintiuno" to 21,
    "veintiun" to 21, "veintiuna" to 21, "veintidos" to 22, "veintitres" to 23, "veinticuatro" to 24, "veinticinco" to 25,
    "veintiseis" to 26, "veintisiete" to 27, "veintiocho" to 28, "veintinueve" to 29)
  private val ES_TENS = mapOf("treinta" to 3, "cuarenta" to 4, "cincuenta" to 5, "sesenta" to 6, "setenta" to 7,
    "ochenta" to 8, "noventa" to 9)
  private val ES_HUNDREDS = mapOf("cien" to 1, "ciento" to 1, "doscientos" to 2, "doscientas" to 2, "trescientos" to 3,
    "trescientas" to 3, "cuatrocientos" to 4, "cuatrocientas" to 4, "quinientos" to 5, "quinientas" to 5,
    "seiscientos" to 6, "seiscientas" to 6, "setecientos" to 7, "setecientas" to 7, "ochocientos" to 8, "ochocientas" to 8,
    "novecientos" to 9, "novecientas" to 9)
  private val ONES2 = ONES - "o"

  /** [words] with the accents dropped first (é -> e). */
  fun words2(text: String): List<String> =
    words(java.text.Normalizer.normalize(text.lowercase(), java.text.Normalizer.Form.NFKD).replace(Regex("\\p{M}+"), ""))

  /** [digitStream] for Spanish too: the English words but "o", and the Spanish number words. */
  fun digitStream2(text: String): String {
    val w = words2(text)
    val out = StringBuilder()
    fun one(j: Int): Int? {
      val t = w.getOrNull(j) ?: return null
      return ONES2[t] ?: ES_ONES[t] ?: t.takeIf { it.length == 1 && it[0].isDigit() }?.toInt()
    }
    fun teen(t: String): Int? = TEENS[t] ?: ES_TEENS[t]
    fun tens(j: Int): Pair<String, Int> {  // a tens word at j (+ "y" + 1..9 / a 1..9 word): its two digits, the next index
      val t = w[j]
      val d = TENS[t] ?: ES_TENS.getValue(t)
      val k = j + 1
      if (t in ES_TENS && w.getOrNull(k) == "y") {
        val o = one(k + 1)
        if (o != null && o > 0) return "$d$o" to k + 2
      }
      if (t in TENS) {
        val o = w.getOrNull(k)?.let { ONES2[it] }
        if (o != null && o > 0) return "$d$o" to k + 1
      }
      return "${d}0" to k
    }
    var i = 0
    while (i < w.size) {
      val t = w[i]
      when {
        t.all { it.isDigit() } -> { out.append(t); i++ }
        t in ONES2 || t in ES_ONES -> { out.append(one(i)); i++ }
        teen(t) != null -> { out.append(teen(t)); i++ }
        t in TENS || t in ES_TENS -> { val (s, n) = tens(i); out.append(s); i = n }
        (t == "double" || t == "triple" || t == "doble") && one(i + 1) != null -> {
          repeat(if (t == "triple") 3 else 2) { out.append(one(i + 1)) }
          i += 2
        }
        t == "hundred" || t in ES_HUNDREDS -> {
          ES_HUNDREDS[t]?.let { out.append(it) }
          var j = i + 1
          if (w.getOrNull(j) == "and" || w.getOrNull(j) == "y") j++
          val n = w.getOrNull(j)
          when {
            n != null && teen(n) != null -> { out.append(teen(n)); i = j + 1 }
            n != null && (n in TENS || n in ES_TENS) -> { val (s, k) = tens(j); out.append(s); i = k }
            n != null && (one(j) ?: 0) > 0 -> { out.append(0).append(one(j)); i = j + 1 }
            else -> { out.append("00"); i++ }
          }
        }
        else -> i++
      }
    }
    return out.toString()
  }

  fun hasNumber2(text: String, number: String): Boolean = digitStream2(text).contains(number.filter { it.isDigit() })

  fun namesCaller2(text: String, caller: String): Boolean {
    val have = words2(text).toSet()
    return words2(caller).all { it in have }
  }
}
