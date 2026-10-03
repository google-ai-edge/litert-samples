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

// Vendored from https://huggingface.co/litert-community/Laya-Multilingual-LiteRT/blob/32f1b84d55f42a323464fad22594cfc2059e7467/android/app/src/main/java/com/laya/LayaJson.kt (Apache-2.0)
// formatted for this repository's 100-column and brace rules; no logic change
// SPDX-License-Identifier: Apache-2.0
package com.google.ai.edge.examples.litert_model_zoo.models.typed_decisions.laya

import java.io.File
import java.io.Reader
import java.io.StringReader
import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode

/** JSON without Android dependencies. Object insertion order is part of the Laya host contract. */
object LayaJson {
  /** Reads UTF-8 JSON and closes the owned file reader. */
  fun parse(file: File): Any? = file.reader(Charsets.UTF_8).use { Parser(it).parse() }

  /** Parses a complete JSON value while preserving object insertion order. */
  fun parse(text: String): Any? = StringReader(text).use { Parser(it).parse() }

  /** Parses a caller-owned reader without closing it. */
  fun parse(reader: Reader): Any? = Parser(reader).parse()

  @Suppress("UNCHECKED_CAST")
  /** Requires an insertion-ordered object produced by the parser. */
  fun asObject(value: Any?): Map<String, Any?> =
    value as? Map<String, Any?>
      ?: throw IllegalArgumentException(
        "Expected a JSON object, got ${value?.javaClass?.simpleName}"
      )

  @Suppress("UNCHECKED_CAST")
  /** Requires a JSON list without changing its element order. */
  fun asArray(value: Any?): List<Any?> =
    value as? List<Any?>
      ?: throw IllegalArgumentException(
        "Expected a JSON array, got ${value?.javaClass?.simpleName}"
      )

  /** Python json.dumps separators, booleans/null and optional ASCII escaping. */
  fun stringify(value: Any?, ensureAscii: Boolean = false): String = buildString {
    appendValue(value, ensureAscii)
  }

  private fun StringBuilder.appendValue(value: Any?, ascii: Boolean) {
    when (value) {
      null -> append("null")
      is String -> appendQuoted(value, ascii)
      is Boolean -> append(if (value) "true" else "false")
      is Byte,
      is Short,
      is Int,
      is Long,
      is BigInteger -> append(value.toString())
      is Float -> append(pythonFloat(value.toDouble()))
      is Double -> append(pythonFloat(value))
      is BigDecimal -> append(pythonFloat(value.toDouble()))
      is Map<*, *> -> {
        append('{')
        value.entries.forEachIndexed { index, entry ->
          if (index > 0) {
            append(", ")
          }
          val key =
            when (val k = entry.key) {
              is String -> k
              null -> "null"
              is Boolean -> if (k) "true" else "false"
              is Number ->
                if (k is Double || k is Float) pythonFloat(k.toDouble()) else k.toString()
              else ->
                throw IllegalArgumentException("JSON object key must be a string/number/bool/null")
            }
          appendQuoted(key, ascii)
          append(": ")
          appendValue(entry.value, ascii)
        }
        append('}')
      }
      is Iterable<*> -> {
        append('[')
        value.forEachIndexed { index, item ->
          if (index > 0) {
            append(", ")
          }
          appendValue(item, ascii)
        }
        append(']')
      }
      is Array<*> -> appendValue(value.asList(), ascii)
      is IntArray -> appendValue(value.asList(), ascii)
      is LongArray -> appendValue(value.asList(), ascii)
      is FloatArray -> appendValue(value.asList(), ascii)
      is DoubleArray -> appendValue(value.asList(), ascii)
      else -> throw IllegalArgumentException("Unsupported JSON value ${value.javaClass.name}")
    }
  }

  private fun StringBuilder.appendQuoted(value: String, ascii: Boolean) {
    append('"')
    for (c in value) {
      when (c) {
        '"' -> append("\\\"")
        '\\' -> append("\\\\")
        '\b' -> append("\\b")
        '\u000c' -> append("\\f")
        '\n' -> append("\\n")
        '\r' -> append("\\r")
        '\t' -> append("\\t")
        else ->
          if (c.code < 32 || (ascii && c.code > 126)) {
            append("\\u")
            append(c.code.toString(16).padStart(4, '0'))
          } else {
            append(c)
          }
      }
    }
    append('"')
  }

  /** Shortest round-tripping decimal, rendered with CPython's fixed/scientific boundaries. */
  private fun pythonFloat(value: Double): String {
    if (value.isNaN()) return "NaN"
    if (value == Double.POSITIVE_INFINITY) return "Infinity"
    if (value == Double.NEGATIVE_INFINITY) return "-Infinity"
    if (value == 0.0) return if (java.lang.Double.doubleToRawLongBits(value) < 0) "-0.0" else "0.0"
    val exact = BigDecimal(value)
    var shortest = exact
    for (precision in 1..17) {
      val candidate =
        exact.round(MathContext(precision, RoundingMode.HALF_EVEN)).stripTrailingZeros()
      if (candidate.toDouble() == value) {
        shortest = candidate
        break
      }
    }
    val exponent = shortest.precision() - shortest.scale() - 1
    if (exponent < -4 || exponent >= 16) {
      val digits = shortest.unscaledValue().abs().toString()
      return buildString {
        if (value < 0) {
          append('-')
        }
        append(digits[0])
        if (digits.length > 1) {
          append('.').append(digits.substring(1))
        }
        append('e')
        append(if (exponent < 0) '-' else '+')
        append(kotlin.math.abs(exponent).toString().padStart(2, '0'))
      }
    }
    return shortest.toPlainString().let { if ('.' in it) it else "$it.0" }
  }

  /** A bounded character buffer avoids keeping a second full 34 MB tokenizer string in memory. */
  private class Parser(private val reader: Reader) {
    private val buffer = CharArray(32 * 1024)
    private var cursor = 0
    private var limit = 0
    private var offset = 0L

    private fun peek(): Int {
      if (limit < 0) return -1
      if (cursor == limit) {
        limit = reader.read(buffer)
        cursor = 0
        if (limit < 0) return -1
      }
      return buffer[cursor].code
    }

    private fun take(): Int =
      peek().also {
        if (it >= 0) {
          cursor++
          offset++
        }
      }

    private fun fail(message: String): Nothing =
      throw IllegalArgumentException("$message at JSON offset $offset")

    private fun whitespace() {
      while (true) {
        when (peek()) {
          32,
          9,
          10,
          13 -> take()
          else -> return
        }
      }
    }

    fun parse(): Any? {
      whitespace()
      val value = value()
      whitespace()
      if (peek() != -1) {
        fail("Trailing content")
      }
      return value
    }

    private fun value(): Any? {
      whitespace()
      return when (peek()) {
        '"'.code -> string()
        '{'.code -> objectValue()
        '['.code -> arrayValue()
        't'.code -> {
          literal("true")
          true
        }
        'f'.code -> {
          literal("false")
          false
        }
        'n'.code -> {
          literal("null")
          null
        }
        'N'.code -> {
          literal("NaN")
          Double.NaN
        }
        'I'.code -> {
          literal("Infinity")
          Double.POSITIVE_INFINITY
        }
        '-'.code,
        in '0'.code..'9'.code -> number()
        else -> fail("Expected a JSON value")
      }
    }

    private fun literal(expected: String) {
      expected.forEach {
        if (take() != it.code) {
          fail("Expected $expected")
        }
      }
    }

    private fun objectValue(): Map<String, Any?> {
      take()
      val out = LinkedHashMap<String, Any?>()
      whitespace()
      if (peek() == '}'.code) {
        take()
        return out
      }
      while (true) {
        whitespace()
        if (peek() != '"'.code) {
          fail("Expected object key")
        }
        val key = string()
        whitespace()
        if (take() != ':'.code) {
          fail("Expected colon")
        }
        out[key] = value()
        whitespace()
        when (take()) {
          '}'.code -> return out
          ','.code -> Unit
          else -> fail("Expected comma or closing brace")
        }
      }
    }

    private fun arrayValue(): List<Any?> {
      take()
      val out = ArrayList<Any?>()
      whitespace()
      if (peek() == ']'.code) {
        take()
        return out
      }
      while (true) {
        out.add(value())
        whitespace()
        when (take()) {
          ']'.code -> return out
          ','.code -> Unit
          else -> fail("Expected comma or closing bracket")
        }
      }
    }

    private fun string(): String {
      if (take() != '"'.code) {
        fail("Expected a string")
      }
      val out = StringBuilder()
      while (true) {
        when (val c = take()) {
          -1 -> fail("Unterminated string")
          '"'.code -> return out.toString()
          '\\'.code ->
            when (val escaped = take()) {
              '"'.code,
              '\\'.code,
              '/'.code -> out.append(escaped.toChar())
              'b'.code -> out.append('\b')
              'f'.code -> out.append('\u000c')
              'n'.code -> out.append('\n')
              'r'.code -> out.append('\r')
              't'.code -> out.append('\t')
              'u'.code -> {
                var code = 0
                repeat(4) {
                  val digit = take().toChar().digitToIntOrNull(16) ?: fail("Invalid Unicode escape")
                  code = (code shl 4) or digit
                }
                out.append(code.toChar())
              }
              else -> fail("Invalid string escape")
            }
          in 0..31 -> fail("Unescaped control character")
          else -> out.append(c.toChar())
        }
      }
    }

    private fun number(): Number {
      val text = StringBuilder()
      if (peek() == '-'.code) {
        text.append(take().toChar())
      }
      if (peek() == 'I'.code) {
        literal("Infinity")
        return Double.NEGATIVE_INFINITY
      }
      if (peek() == '0'.code) {
        text.append(take().toChar())
      } else {
        if (peek() !in '1'.code..'9'.code) {
          fail("Invalid number")
        }
        while (peek() in '0'.code..'9'.code) {
          text.append(take().toChar())
        }
      }
      var fractional = false
      if (peek() == '.'.code) {
        fractional = true
        text.append(take().toChar())
        if (peek() !in '0'.code..'9'.code) {
          fail("Invalid fraction")
        }
        while (peek() in '0'.code..'9'.code) {
          text.append(take().toChar())
        }
      }
      if (peek() == 'e'.code || peek() == 'E'.code) {
        fractional = true
        text.append(take().toChar())
        if (peek() == '+'.code || peek() == '-'.code) {
          text.append(take().toChar())
        }
        if (peek() !in '0'.code..'9'.code) {
          fail("Invalid exponent")
        }
        while (peek() in '0'.code..'9'.code) {
          text.append(take().toChar())
        }
      }
      val token = text.toString()
      return if (fractional) token.toDouble() else token.toLongOrNull() ?: BigInteger(token)
    }
  }
}
