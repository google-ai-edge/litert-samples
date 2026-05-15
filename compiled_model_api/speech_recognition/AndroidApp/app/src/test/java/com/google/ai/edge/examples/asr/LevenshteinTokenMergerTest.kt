/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class LevenshteinTokenMergerTest {

  class MockTokenizer(override val vocabSize: Int = 100) : Tokenizer {

    private val tokenMap = mutableMapOf<Int, String>()
    var nextId = 1

    fun addWord(word: String): IntArray {
      val ids = mutableListOf<Int>()
      for (char in word) {
        val id = nextId++
        tokenMap[id] = char.toString()
        ids.add(id)
      }
      return ids.toIntArray()
    }

    fun addSpace(): IntArray {
      val id = nextId++
      tokenMap[id] = " "
      return intArrayOf(id)
    }

    override fun decode(tokenIds: IntArray): String {
      return tokenIds.map { tokenMap[it] ?: "" }.joinToString("")
    }

    override fun close() {}
  }

  private fun decodeAll(
    merger: LevenshteinTokenMerger,
    tokens: IntArray,
    decodeEndOfSequence: Boolean = true,
    noTimestamps: Boolean = true,
    startTimestamp: Int = 0,
  ): Postprocessor.DecodedText {
    val confirmed = StringBuilder()
    var lastUnconfirmed = ""
    for ((i, t) in tokens.withIndex()) {
      val result =
        merger.decode(t, if (noTimestamps) SpeechRecognizer.NO_TIMESTAMP else startTimestamp + i)
      if (result != null) {
        if (result.confirmedText.isNotEmpty()) {
          if (confirmed.isNotEmpty()) confirmed.append(" ")
          confirmed.append(result.confirmedText)
        }
        lastUnconfirmed = result.unconfirmedText
      }
    }
    if (decodeEndOfSequence) {
      val finalResult =
        merger.decode(
          SpeechRecognizer.END_OF_SEQUENCE,
          if (noTimestamps) SpeechRecognizer.NO_TIMESTAMP else startTimestamp + tokens.size,
        )
      if (finalResult != null) {
        if (finalResult.confirmedText.isNotEmpty()) {
          if (confirmed.isNotEmpty()) confirmed.append(" ")
          confirmed.append(finalResult.confirmedText)
        }
        if (finalResult.unconfirmedText.isNotEmpty()) {
          lastUnconfirmed = finalResult.unconfirmedText
        }
      }
    }
    return Postprocessor.DecodedText(confirmed.toString(), lastUnconfirmed)
  }

  @Test
  fun testDecode_noOverlap() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.0f)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")

    val (confirmed, unconfirmed) = decodeAll(merger, tokens1)

    assertEquals("", confirmed)
    assertEquals("hello world", unconfirmed)
  }

  @Test
  fun testDecode_withOverlap_perfectAlign() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    val (conf1, unconf1) = decodeAll(merger, tokens1)
    assertEquals("", conf1)
    assertEquals("hello world how", unconf1)

    val tokens2 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello", conf2)
    assertEquals("world how are", unconf2)
  }

  @Test
  fun testDecode_withOverlap_customPivotFactor() {
    val tokenizer = MockTokenizer()
    val merger =
      LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 2, pivotFactor = 0.2f)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello", conf2)
    assertEquals("world how are", unconf2)
  }

  @Test
  fun testDecode_withOverlap_customSearchWindow() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 5)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello", conf2)
    assertEquals("world how are", unconf2)
  }

  @Test
  fun testDecode_dedupOverlapWord() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 2)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello", conf2)
    assertEquals("world how are", unconf2)
  }

  @Test
  fun testDecode_actualRepeatedWord_notDeduped() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 2)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace() +
        tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello", conf2)
    assertEquals("world how how are", unconf2)
  }

  private fun addWords(tokenizer: MockTokenizer, text: String): IntArray {
    val tokens = mutableListOf<Int>()
    val words = text.split(" ")
    for ((index, word) in words.withIndex()) {
      tokens.addAll(tokenizer.addWord(word).toList())
      if (index < words.size - 1) {
        tokens.addAll(tokenizer.addSpace().toList())
      }
    }
    return tokens.toIntArray()
  }

  @Test
  fun testDecode_realWorldExampleBeforeEndOfSequence() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.8f)

    val text1 = "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes."
    val text2 = "Wish to see it anymore, observed Phoebe, turning"

    val (conf1, unconf1) = decodeAll(merger, addWords(tokenizer, text1))
    val (conf2, unconf2) =
      decodeAll(merger, addWords(tokenizer, text2), decodeEndOfSequence = false)

    val idealFinalMergedWords =
      "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes."

    val actualMerged =
      listOf(conf1, unconf1, conf2, unconf2).filter { it.isNotEmpty() }.joinToString(" ")
    assertEquals(idealFinalMergedWords, actualMerged)
  }

  @Test
  fun testDecode_realWorldExampleWithEndOfSequence() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.8f)

    val text1 = "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes."
    val text2 =
      "Wish to see it anymore, observed Phoebe, turning away her eyes. It is certainly very, very."
    val text3 =
      "Or observe Phoebe, turning away her eyes. It is certainly very like the old portrait."

    val (conf1, unconf1) = decodeAll(merger, addWords(tokenizer, text1))
    val (conf2, unconf2) = decodeAll(merger, addWords(tokenizer, text2))

    val idealIntermediateMergedWords =
      "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is " +
        "certainly very, very."

    val actualIntermediateMerged =
      listOf(conf1, conf2, unconf2).filter { it.isNotEmpty() }.joinToString(" ")
    assertEquals(idealIntermediateMergedWords, actualIntermediateMerged)

    val (conf3, unconf3) = decodeAll(merger, addWords(tokenizer, text3))

    val idealFinalMergedWords =
      "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is " +
        "certainly very like the old portrait."

    val actualMerged =
      listOf(conf1, conf2, conf3, unconf3).filter { it.isNotEmpty() }.joinToString(" ")

    assertEquals(idealFinalMergedWords, actualMerged)
  }

  @Test
  fun testDecode_incrementalOutput() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    val (conf1, unconf1) = decodeAll(merger, tokens1)
    assertEquals("", conf1)
    assertEquals("hello world how", unconf1)

    val part1 =
      tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how") +
        tokenizer.addSpace()
    var confirmed = ""
    for (t in part1) {
      val res = merger.decode(t, SpeechRecognizer.NO_TIMESTAMP)
      if (res != null && res.confirmedText.isNotEmpty()) {
        confirmed = res.confirmedText
      }
    }
    assertEquals("hello", confirmed)

    val part2 = tokenizer.addWord("are")
    for (t in part2) {
      val res = merger.decode(t, SpeechRecognizer.NO_TIMESTAMP)
      if (res != null && res.confirmedText.isNotEmpty()) {
        confirmed = res.confirmedText
      }
    }
    assertEquals("hello", confirmed)

    val finalResult = merger.decode(SpeechRecognizer.END_OF_SEQUENCE, SpeechRecognizer.NO_TIMESTAMP)
    assertTrue(finalResult == null)
  }

  private fun getOverlap(list1: List<String>, list2: List<String>): Int {
    val maxSize = minOf(list1.size, list2.size)
    for (size in maxSize downTo 1) {
      if (list1.takeLast(size) == list2.take(size)) {
        return size
      }
    }
    return 0
  }

  @Test
  fun testMergeIntoUnconfirmedText_noOverlap() {
    val prevWords = listOf("hello")
    val currentWords = listOf("world")
    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = emptyList(),
        currentWords = currentWords,
        currentWordTimestamps = emptyList(),
        overlapRatio = 0.5f,
        pivotFactor = 0.5f,
        prevWordIndexOfUnconfirmed = 0,
        prevWordIndexOfPivot = 0,
      )
    assertEquals(listOf("world"), result)
  }

  @Test
  fun testMergeIntoUnconfirmedText_perfectOverlap() {
    val prevWords = listOf("hello", "world")
    val currentWords = listOf("world", "how")
    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = emptyList(),
        currentWords = currentWords,
        currentWordTimestamps = emptyList(),
        overlapRatio = 0.5f,
        pivotFactor = 0.5f,
        prevWordIndexOfUnconfirmed = 1,
        prevWordIndexOfPivot = 1,
      )
    assertEquals(listOf("world", "how"), result)
  }

  @Test
  fun testMergeIntoUnconfirmedText_shortCurrentWords() {
    val prevWords =
      "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.".split(" ")
    val currentWords = "Wish to see it anymore, observed Phoebe, turning".split(" ")
    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = emptyList(),
        currentWords = currentWords,
        currentWordTimestamps = emptyList(),
        overlapRatio = 0.8f,
        pivotFactor = 0.6f,
        prevWordIndexOfUnconfirmed = 3,
        prevWordIndexOfPivot = 10,
      )
    assertEquals(
      "wish to see it any more, observed Phoebe, turning away her eyes.".split(" "),
      result,
    )
  }

  @Test
  fun testMergeIntoUnconfirmedText_fullCurrentWords() {
    val prevWords =
      "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.".split(" ")
    val currentWords =
      "Wish to see it anymore, observed Phoebe, turning away her eyes. It is certainly very, very."
        .split(" ")
    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = emptyList(),
        currentWords = currentWords,
        currentWordTimestamps = emptyList(),
        overlapRatio = 0.8f,
        pivotFactor = 0.6f,
        prevWordIndexOfUnconfirmed = 3,
        prevWordIndexOfPivot = 10,
      )
    assertEquals(
      "wish to see it any more, observed turning away her eyes. It is certainly very, very."
        .split(" "),
      result,
    )
  }

  @Test
  fun testMergeIntoUnconfirmedText_withTimestamps_shiftsPivotWordsDynamically() {
    val prevWords = listOf("a", "b", "c", "d", "e")
    val prevTimestamps = listOf(10, 20, 30, 40, 50)
    val currentWords = listOf("c", "d", "e", "f", "g", "h")

    // We make the first 4 words in currentWords very fast (timestamps 25, 26, 27, 28).
    // The next words are slower.
    // prevWordIndexOfPivot is 3 (so timestampAtPivot = prevTimestamps[3] = 40).
    // currentWordTimestamps under 40 are all the first 4! (25, 26, 27, 28).
    // So numWordsBeforePivotInCurrent = 4.
    val currentTimestamps = listOf(25, 26, 27, 28, 45, 55)

    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = prevTimestamps,
        currentWords = currentWords,
        currentWordTimestamps = currentTimestamps,
        overlapRatio = 0.5f,
        pivotFactor = 0.5f,
        prevWordIndexOfUnconfirmed = 2, // "c"
        prevWordIndexOfPivot = 3, // "d"
      )

    // prevWords from 2 to 3 is ["c"]
    // currentWords from 4 (exclusive of first 4 before pivot) is ["g", "h"]
    // So output should be ["c", "g", "h"]
    assertEquals(listOf("c", "g", "h"), result)
  }

  @Test
  fun testMergeIntoUnconfirmedText_withTimestamps_pivotExceedsPrevWordsSize() {
    val prevWords = listOf("foo", "bar")
    val prevTimestamps = listOf(0, 100)
    val currentWords = listOf("bar", "baz", "qux")
    val currentTimestamps = listOf(100, 101, 150)

    val result =
      LevenshteinTokenMerger.mergeIntoUnconfirmedText(
        prevWords = prevWords,
        prevWordTimestamps = prevTimestamps,
        currentWords = currentWords,
        currentWordTimestamps = currentTimestamps,
        overlapRatio = 0.5f,
        pivotFactor = 0.5f,
        prevWordIndexOfUnconfirmed = 1,
        prevWordIndexOfPivot =
          2, // Evaluates to size! timestampAtPivot becomes prevTimestamps.last() + 1 = 101
      )

    // timestampAtPivot = 101. Current words timestamps under 101: 100 (size 1).
    // unconfirmedTextInPrev = prevWords[1..2] (coerced bounds from size 2) = ["bar"]
    // numWordsBeforePivotInCurrent = 1. currentWords[1..] = ["baz", "qux"]
    // Output dedup("bar", "baz qux") = "bar baz qux"
    assertEquals(listOf("bar", "baz", "qux"), result)
  }

  @Test
  fun testDedupWords_noOverlap() {
    val prevWords = listOf("hello")
    val currentWords = listOf("world")
    val result = LevenshteinTokenMerger.dedupWords(prevWords, currentWords)
    assertEquals(listOf("world"), result)
  }

  @Test
  fun testDedupWords_withOverlap() {
    val prevWords = listOf("hello", "world")
    val currentWords = listOf("world", "how")
    val result = LevenshteinTokenMerger.dedupWords(prevWords, currentWords)
    assertEquals(listOf("how"), result)
  }

  @Test
  fun testDedupWords_withLongerOverlap() {
    val prevWords = listOf("hello", "world", "how")
    val currentWords = listOf("world", "how", "are")
    val result = LevenshteinTokenMerger.dedupWords(prevWords, currentWords)
    assertEquals(listOf("are"), result)
  }

  @Test
  fun testCanonicalize_empty() {
    val result = LevenshteinTokenMerger.canonicalize(emptyList())
    assertEquals("", result)
  }

  @Test
  fun testCanonicalize_standard() {
    val result = LevenshteinTokenMerger.canonicalize(listOf("Hello", "World"))
    assertEquals("hello world", result)
  }

  @Test
  fun testCanonicalize_withPunctuation() {
    val result = LevenshteinTokenMerger.canonicalize(listOf("Hello,", "world!"))
    assertEquals("hello world", result)
  }

  @Test
  fun testDecode_pivotReachesPrevWordsSize() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    decodeAll(merger, tokens1)

    val tokens2 = tokenizer.addWord("world") + tokenizer.addSpace() + tokenizer.addWord("are")
    val (conf2, unconf2) = decodeAll(merger, tokens2)

    assertEquals("hello", conf2)
    assertEquals("world are", unconf2)
  }

  @Test
  fun testDecode_levenshteinDistance_belowThreshold() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 = tokenizer.addWord("word") + tokenizer.addSpace() + tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello world", conf2)
    assertEquals("how word are", unconf2)
  }

  @Test
  fun testDecode_levenshteinDistance_aboveOrEqualThreshold() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 =
      tokenizer.addWord("hello") +
        tokenizer.addSpace() +
        tokenizer.addWord("world") +
        tokenizer.addSpace() +
        tokenizer.addWord("how")

    decodeAll(merger, tokens1)

    val tokens2 = tokenizer.addWord("abcde") + tokenizer.addSpace() + tokenizer.addWord("are")

    val (conf2, unconf2) = decodeAll(merger, tokens2)
    assertEquals("hello world how", conf2)
    assertEquals("abcde are", unconf2)
  }

  @Test
  fun testDecode_silenceConfirmAllPreviousWords() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    decodeAll(merger, tokens1)

    val finalResult = merger.decode(SpeechRecognizer.END_OF_SEQUENCE, SpeechRecognizer.NO_TIMESTAMP)
    assertNotNull(finalResult)
  }

  @Test
  fun testDecode_removeDuplicatesWithAlreadyConfirmedText() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    // Sequence 1: confirms "hello world" at the end of the sequence.
    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    decodeAll(merger, tokens1)
    merger.decode(SpeechRecognizer.END_OF_SEQUENCE, SpeechRecognizer.NO_TIMESTAMP)

    // Sequence 2: receives "world how" where "world" is a duplicate of already confirmed text.
    val tokens2 = tokenizer.addWord("world") + tokenizer.addSpace() + tokenizer.addWord("how")
    val (conf, unconf) = decodeAll(merger, tokens2)

    // The duplicate word "world" should be removed, leaving only "how" in unconfirmed text.
    assertEquals("", conf)
    assertEquals("how", unconf)
  }

  @Test
  fun testDecode_withTimestamps() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.7f, searchWindow = 0)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    val (conf1, unconf1) = decodeAll(merger, tokens1, noTimestamps = false)
    assertEquals("", conf1)
    assertEquals("hello world", unconf1)

    val tokens2 = tokenizer.addWord("world") + tokenizer.addSpace() + tokenizer.addWord("how")
    val (conf2, unconf2) = decodeAll(merger, tokens2, noTimestamps = false)
    assertEquals("hello", conf2)
    assertEquals("world how", unconf2)
  }

  @Test
  fun testDecode_withTimestamps_dropUnmatchedTokens() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.7f, searchWindow = 0)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    val (conf1, unconf1) = decodeAll(merger, tokens1, noTimestamps = false)
    assertEquals("", conf1)
    assertEquals("hello world", unconf1)

    val tokens2 = tokenizer.addWord("unmatched") + tokenizer.addSpace() + tokenizer.addWord("how")
    val (conf2, unconf2) = decodeAll(merger, tokens2, noTimestamps = false)
    assertEquals("hello world", conf2)
    assertEquals("how", unconf2)
  }

  @Test
  fun testDecode_withTimestampsWithLessOverlap_keepUnmatchedTokens() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.3f, searchWindow = 0)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    val (conf1, unconf1) = decodeAll(merger, tokens1, noTimestamps = false)
    assertEquals("", conf1)
    assertEquals("hello world", unconf1)

    val tokens2 = tokenizer.addWord("unmatched") + tokenizer.addSpace() + tokenizer.addWord("how")
    val (conf2, unconf2) = decodeAll(merger, tokens2, noTimestamps = false)
    assertEquals("hello world", conf2)
    assertEquals("unmatched how", unconf2)
  }

  @Test
  fun testDecode_withTimestamps_searchWindowFindsShiftedWords() {
    val tokenizer = MockTokenizer()
    // A small tolerance (searchWindow = 1) to find the word even if the timestamp alignment isn't
    // perfect.
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 1)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    val (conf1, unconf1) = decodeAll(merger, tokens1, noTimestamps = false)
    assertEquals("", conf1)
    assertEquals("hello world", unconf1)

    // "hello" is skipped, word starts slightly shifted relative to expected overlap.
    // The searchWindow should correctly catch "world".
    val tokens2 = tokenizer.addWord("world") + tokenizer.addSpace() + tokenizer.addWord("how")
    val (conf2, unconf2) = decodeAll(merger, tokens2, noTimestamps = false)
    assertEquals("hello", conf2)
    assertEquals("world how", unconf2)
  }

  @Test
  fun testDecode_withTimestamps_multipleConsecutiveSequences() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f, searchWindow = 0)

    val tokens1 = tokenizer.addWord("one") + tokenizer.addSpace() + tokenizer.addWord("two")
    val (conf1, unconf1) = decodeAll(merger, tokens1, noTimestamps = false)
    assertEquals("", conf1)
    assertEquals("one two", unconf1)

    val tokens2 = tokenizer.addWord("two") + tokenizer.addSpace() + tokenizer.addWord("three")
    val (conf2, unconf2) = decodeAll(merger, tokens2, noTimestamps = false)
    assertEquals("one", conf2)
    assertEquals("two three", unconf2)

    val tokens3 = tokenizer.addWord("three") + tokenizer.addSpace() + tokenizer.addWord("four")
    val (conf3, unconf3) = decodeAll(merger, tokens3, noTimestamps = false)
    assertEquals("two", conf3)
    assertEquals("three four", unconf3)
  }

  @Test
  fun testDecode_withTimestamps_shortCurrentSequence_returnsNull() {
    val tokenizer = MockTokenizer()
    val merger = LevenshteinTokenMerger(tokenizer, overlapRatio = 0.5f)

    val tokens1 = tokenizer.addWord("hello") + tokenizer.addSpace() + tokenizer.addWord("world")
    decodeAll(merger, tokens1, noTimestamps = false, startTimestamp = 0)

    val tokens2 = tokenizer.addWord("world") + tokenizer.addSpace() + tokenizer.addWord("how")
    // Pass a negative startTimestamp so that each token in the sequence has a timestamp <
    // prevWordTimestamps.last().
    // Use decodeEndOfSequence = false as EoS is processed separately.
    val (conf2, unconf2) =
      decodeAll(
        merger,
        tokens2,
        decodeEndOfSequence = false,
        noTimestamps = false,
        startTimestamp = -20,
      )
    assertEquals("", conf2)
    assertEquals("", unconf2)
  }

  @Test
  fun testCloseEnough_String_exactMatch_short() {
    assertTrue(LevenshteinTokenMerger.closeEnough("the", "the"))
  }

  @Test
  fun testCloseEnough_String_exactMatch_long() {
    assertTrue(LevenshteinTokenMerger.closeEnough("hello", "hello"))
  }

  @Test
  fun testCloseEnough_String_different_short() {
    assertFalse(LevenshteinTokenMerger.closeEnough("the", "thy"))
  }

  @Test
  fun testCloseEnough_String_different_long_dist1() {
    assertTrue(LevenshteinTokenMerger.closeEnough("hello", "hellp"))
  }

  @Test
  fun testCloseEnough_String_different_long_dist2() {
    org.junit.Assert.assertFalse(LevenshteinTokenMerger.closeEnough("hello", "helpp"))
  }

  @Test
  fun testCloseEnough_String_prefixMatch() {
    assertTrue(LevenshteinTokenMerger.closeEnough("hello", "hello world"))
  }

  @Test
  fun testCloseEnough_List_same() {
    assertTrue(
      LevenshteinTokenMerger.closeEnough(listOf("hello", "world"), listOf("hello", "world"))
    )
  }

  @Test
  fun testCloseEnough_List_differentLength() {
    org.junit.Assert.assertFalse(
      LevenshteinTokenMerger.closeEnough(listOf("hello"), listOf("hello", "world"))
    )
  }

  @Test
  fun testCloseEnough_List_closeEnoughElements() {
    assertTrue(
      LevenshteinTokenMerger.closeEnough(listOf("hello", "world"), listOf("hellp", "world"))
    )
  }
}
