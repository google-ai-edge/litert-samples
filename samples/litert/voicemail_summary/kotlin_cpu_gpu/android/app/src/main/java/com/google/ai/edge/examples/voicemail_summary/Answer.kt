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
 * The language of the answer. LFM3-Audio-2B speaks the languages of its model card ("Speech languages / locales"); the
 * text-to-speech prompt names the locale (`Perform TTS in <locale> using a <gender> voice.`).
 */
data class AnswerLanguage(val name: String, val ttsLocale: String, val readNumber: ((String) -> String)? = null)

object Answer {
  val LANGUAGES = listOf(
    AnswerLanguage("English", "American English"), AnswerLanguage("Spanish", "Spanish", SpanishNumbers::phone), AnswerLanguage("French", "French"),
    AnswerLanguage("German", "German"), AnswerLanguage("Italian", "Italian"), AnswerLanguage("Portuguese", "Portuguese"),
    AnswerLanguage("Dutch", "Dutch"), AnswerLanguage("Danish", "Danish"), AnswerLanguage("Swedish", "Swedish"),
    AnswerLanguage("Finnish", "Finnish"), AnswerLanguage("Polish", "Polish"), AnswerLanguage("Greek", "Greek"),
    AnswerLanguage("Turkish", "Turkish"), AnswerLanguage("Arabic", "Arabic"), AnswerLanguage("Hebrew", "Hebrew"),
    AnswerLanguage("Japanese", "Japanese"), AnswerLanguage("Korean", "Korean"), AnswerLanguage("Chinese", "Chinese"),
  )

  /** The default instruction for an English answer. */
  const val ENGLISH_INSTRUCTION =
    "Summarize this voicemail in one or two short sentences: who called, what they want, and the number to call back."

  /** The default instruction for an answer in another language (%s = the language's English name). */
  const val OTHER_INSTRUCTION =
    "Listen to this voicemail and answer in %s: who called, what they want, and the number to call back."

  /** A Spanish answer's own default instruction, when it is not OTHER_INSTRUCTION; null = OTHER_INSTRUCTION. */
  val SPANISH_INSTRUCTION: String? = null

  /**
   * How an answer in another language than English is made: `interleaved` (one pass of the model's voice chat: text
   * and speech in turns) or `two_stage` (the same model twice: the answer as text, then that text spoken by its
   * text-to-speech). English answers are interleaved.
   */
  const val OTHER_PATH = "two_stage"

  fun instruction(lang: AnswerLanguage): String = when {
    lang.name == "English" -> ENGLISH_INSTRUCTION
    lang.name == "Spanish" && SPANISH_INSTRUCTION != null -> SPANISH_INSTRUCTION
    else -> OTHER_INSTRUCTION.format(lang.name)
  }

  fun path(lang: AnswerLanguage): String = if (lang.name == "English") "interleaved" else OTHER_PATH

  /**
   * The text the text-to-speech stage of `two_stage` reads, for a language that has a way to read phone numbers (only
   * Spanish so far; every other language gets the answer as written): the answer without its markdown (bold, headings,
   * list marks, line breaks) and with each phone number in words, because the model's text to speech garbled digit
   * strings, and repeated digit words, in the development runs. The screen keeps the answer as written.
   */
  fun speakable(text: String, lang: AnswerLanguage): String {
    val read = lang.readNumber ?: return text
    val plain = text.replace(Regex("""\*\*|__|`"""), "")
      .replace(Regex("""(?m)^\s*(?:#{1,6}\s+|[-*\u2022]\s+|\d+[.)]\s+)"""), "")
      .replace(Regex("""\s*\n+\s*"""), " ").trim()
    return Callback.replaceNumbers(plain, read)
  }

  fun byName(name: String): AnswerLanguage? = LANGUAGES.firstOrNull { it.name.equals(name, ignoreCase = true) }
}

/**
 * A phone number as a Spanish speaker says it, in the groups it is written in, four digits as two pairs, each group
 * as one number after its leading zeros: "555-0161" -> "quinientos cincuenta y cinco, cero uno, sesenta y uno".
 */
object SpanishNumbers {
  private val ONES = listOf("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve")
  private val TEENS = listOf("diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve")
  private val TWENTIES = listOf("veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis",
    "veintisiete", "veintiocho", "veintinueve")
  private val TENS = listOf("", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa")
  private val HUNDREDS = listOf("", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos",
    "setecientos", "ochocientos", "novecientos")

  /** 0..999 in words. */
  fun words(n: Int): String = when {
    n < 10 -> ONES[n]
    n < 20 -> TEENS[n - 10]
    n < 30 -> TWENTIES[n - 20]
    n < 100 -> TENS[n / 10] + if (n % 10 != 0) " y " + ONES[n % 10] else ""
    n == 100 -> "cien"
    else -> HUNDREDS[n / 100] + if (n % 100 != 0) " " + words(n % 100) else ""
  }

  /** One group: its leading zeros as "cero", the rest as one number (digit by digit past three digits). */
  private fun group(g: String): String {
    val zeros = g.takeWhile { it == '0' }
    val rest = g.drop(zeros.length)
    val tail = when {
      rest.isEmpty() -> emptyList()
      rest.length > 3 -> rest.map { ONES[it - '0'] }
      else -> listOf(words(rest.toInt()))
    }
    return (zeros.map { "cero" } + tail).joinToString(" ")
  }

  fun phone(number: String): String = number.split(Regex("""\D+""")).filter { it.isNotEmpty() }.flatMap { g ->
    when (g.length) {
      4 -> listOf(g.substring(0, 2), g.substring(2))
      7 -> listOf(g.substring(0, 3), g.substring(3, 5), g.substring(5))
      10 -> listOf(g.substring(0, 3), g.substring(3, 6), g.substring(6, 8), g.substring(8))
      else -> listOf(g)
    }
  }.joinToString(", ") { group(it) }
}
