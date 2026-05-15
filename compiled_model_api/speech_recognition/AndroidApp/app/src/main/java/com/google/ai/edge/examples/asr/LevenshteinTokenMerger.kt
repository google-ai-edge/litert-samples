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

import android.util.Log
import androidx.annotation.VisibleForTesting
import kotlin.math.ceil

class LevenshteinTokenMerger(
  private val tokenizer: Tokenizer,
  private val overlapRatio: Float,
  private val searchWindow: Int = 2,
  private val maxLevenshteinDistance: Int = 5,
  private val pivotFactor: Float = 0.6f,
) : Postprocessor {
  /**
   * The words (confirmed and unconfirmed) cached to re-calculate the unconfirmed text merged with
   * the current words.
   */
  private var prevWords = listOf<String>()
  /**
   * Timestamps of the words of the corresponding index in prevWords. Note that the timestamps are
   * aligned with the current timestamps.
   */
  private var prevWordTimestamps = listOf<Int>()
  /** Index of the first word in prevWords that is considered as unconfirmed text. */
  private var prevWordIndexOfUnconfirmed = -1
  /**
   * Index of the first word in prevWords that is considered as pivot. Text after
   * prevWordIndexOfUnconfirmed before this point will be unconfirmed text from previous sequence.
   * Rest of unconfirmed text will come from the current words.
   */
  private var prevWordIndexOfPivot = -1
  /** Current sequence of tokens recognized by the model so far. */
  private val currentTokens = mutableListOf<Int>()
  /** Timestamps of the current words interpolated from tokens' timestamps. */
  private val currentWordTimestamps = mutableListOf<Int>()
  /** Last confirmed words */
  private var lastConfirmedWords = listOf<String>()

  override fun close() {}

  override fun decode(tokenId: Int, timestamp: Int): Postprocessor.DecodedText? {
    Log.d(TAG, "decode: tokenId=$tokenId, timestamp=$timestamp")
    val isEndOfSequence = tokenId == SpeechRecognizer.END_OF_SEQUENCE
    if (!isEndOfSequence) {
      currentTokens.add(tokenId)
    }

    val currentWords = detokenize(currentTokens.toIntArray())
    if (timestamp != SpeechRecognizer.NO_TIMESTAMP) {
      // Approximate the timestamps of the words added newly based on token timestamps.
      while (currentWords.size > currentWordTimestamps.size) {
        currentWordTimestamps.add(timestamp)
      }
    }
    Log.d(TAG, "decode: currentWords=$currentWords, currentWordTimestamps=$currentWordTimestamps")

    if (isEndOfSequence) {
      val returnValue = getConfirmedAndUnconfirmedTextAtEndOfSequence(currentWords)
      // Adjust the timestamps in prevWordTimestamps to be aligned with the next sequence.
      if (timestamp != SpeechRecognizer.NO_TIMESTAMP) {
        val timestampStep = (timestamp * (1 - overlapRatio)).toInt()
        prevWordTimestamps = prevWordTimestamps.map { it - timestampStep }
      }
      currentTokens.clear()
      currentWordTimestamps.clear()
      prevWordIndexOfUnconfirmed = -1
      prevWordIndexOfPivot = -1
      Log.d(
        TAG,
        "decode: prevWords=$prevWords, prevWordTimestamps=$prevWordTimestamps, " +
          "returnValue=$returnValue",
      )
      return returnValue
    }

    // Output the current words if this is the first sequence of tokens.
    if (prevWords.isEmpty()) {
      val returnValue = Postprocessor.DecodedText("", currentWords.joinToString(" "))
      Log.d(TAG, "decode: prevWords is empty, returnValue=$returnValue")
      return returnValue
    }

    // If the current sequence of words is not long enough, don't change the previous output.
    if (
      (timestamp != SpeechRecognizer.NO_TIMESTAMP && timestamp < prevWordTimestamps.last()) ||
        (timestamp == SpeechRecognizer.NO_TIMESTAMP &&
          currentWords.size < prevWords.size * overlapRatio)
    ) {
      Log.d(TAG, "decode: currentWords is too short, return null")
      return null
    }

    // The points of confirmation and pivot has already been calculated in the previous call.
    // Just update the unconfirmed text with new tokens.
    if (prevWordIndexOfUnconfirmed != -1) {
      val returnValue =
        Postprocessor.DecodedText("", mergeIntoUnconfirmedText(currentWords).joinToString(" "))
      Log.d(TAG, "decode: prevWordIndexOfUnconfirmed != -1, returnValue=$returnValue")
      return returnValue
    }

    val returnValue = getConfirmedAndUnconfirmedTextFromPrevWordsAndCurrentWords(currentWords)
    Log.d(TAG, "decode: returnValue=$returnValue")
    return returnValue
  }

  private fun getConfirmedAndUnconfirmedTextAtEndOfSequence(
    currentWords: List<String>
  ): Postprocessor.DecodedText? {
    val mergedWords = mergeIntoUnconfirmedText(currentWords)
    val mergedTimestamps = mergeTimestamps(mergedWords.size)
    if (mergedWords.isEmpty() && prevWords.isNotEmpty()) {
      // (Merged) current sequence is empty, i.e. silence. Confirm all the previous words.
      lastConfirmedWords = prevWords
      prevWords = mergedWords
      prevWordTimestamps = mergedTimestamps
      return Postprocessor.DecodedText(lastConfirmedWords.joinToString(" "), "")
    }

    // Dedup the unconfirmed text with the last confirmed words in case when the current
    // sequence includes longer silence than usual.
    prevWords = dedupWords(lastConfirmedWords, mergedWords, mergedWords.size)
    prevWordTimestamps =
      mergedTimestamps.drop((mergedTimestamps.size - prevWords.size).coerceAtLeast(0))
    if (prevWords.isNotEmpty() && prevWordIndexOfUnconfirmed == -1) {
      // Nothing has returned yet for the current sequence. Return it now as all unconfirmed.
      return Postprocessor.DecodedText("", prevWords.joinToString(" "))
    }

    return null
  }

  /**
   * Gets the confirmed and unconfirmed text from the previous words and current words by
   * calculating 2 points in PrevWords:
   * 1) where the current words can be aligned with the previous words using Levenshtein distance,
   *    i.e. the previous words before the point are considered as confirmedText.
   * 2) the pivot point in the previous words where the preivous words before are considered as
   *    unconfirmedText.
   */
  private fun getConfirmedAndUnconfirmedTextFromPrevWordsAndCurrentWords(
    currentWords: List<String>
  ): Postprocessor.DecodedText {
    val middleIndexToSearch =
      if (prevWordTimestamps.isNotEmpty() && currentWordTimestamps.isNotEmpty()) {
        // If available, find the index of the timestamp in prevWordTimestamps that is equal to or
        // larger than the timestamp of the first current word.
        val targetTimestamp = currentWordTimestamps.first()
        prevWordTimestamps
          .indexOfFirst { it >= targetTimestamp }
          .let { index ->
            if (index != -1) index else (prevWords.size * (1 - overlapRatio)).toInt()
          }
      } else {
        (prevWords.size * (1 - overlapRatio)).toInt()
      }
    Log.d(TAG, "decode: middleIndexToSearch=$middleIndexToSearch")
    var minDistance = maxLevenshteinDistance
    // If no overlap is found, set it to the end of the previous words.
    prevWordIndexOfUnconfirmed =
      (middleIndexToSearch + searchWindow + 1).coerceAtMost(prevWords.size)
    for (i in -searchWindow..searchWindow) {
      val startIndex = (middleIndexToSearch + i).coerceIn(prevWords.indices)
      val endIndex = (startIndex + currentWords.size).coerceAtMost(prevWords.size)
      val dist =
        levenshtein(
          canonicalize(prevWords.subList(startIndex, endIndex)),
          canonicalize(
            currentWords.subList(0, (endIndex - startIndex).coerceAtMost(currentWords.size))
          ),
        )
      Log.d(TAG, "decode: dist=$dist, startIndex=$startIndex, endIndex=$endIndex")
      if (dist < minDistance) {
        minDistance = dist
        prevWordIndexOfUnconfirmed = startIndex
        Log.d(
          TAG,
          "Better alignment: dist=$dist, prevWordIndexOfUnconfirmed=$prevWordIndexOfUnconfirmed",
        )
      }
    }
    val unconfirmedWordsUntilPivot =
      ceil((prevWords.size - prevWordIndexOfUnconfirmed) * pivotFactor).toInt()
    prevWordIndexOfPivot =
      (prevWordIndexOfUnconfirmed + unconfirmedWordsUntilPivot).coerceAtMost(prevWords.size)
    lastConfirmedWords = prevWords.subList(0, prevWordIndexOfUnconfirmed)
    return Postprocessor.DecodedText(
      lastConfirmedWords.joinToString(" "),
      mergeIntoUnconfirmedText(currentWords).joinToString(" "),
    )
  }

  private fun detokenize(tokens: IntArray): List<String> =
    tokenizer.decode(tokens).split(" ").map { it.trim() }.filter { it.isNotEmpty() }

  private fun mergeIntoUnconfirmedText(currentWords: List<String>) =
    if (prevWordIndexOfUnconfirmed == -1) {
      currentWords
    } else {
      mergeIntoUnconfirmedText(
        prevWords,
        prevWordTimestamps,
        currentWords,
        currentWordTimestamps,
        overlapRatio,
        pivotFactor,
        prevWordIndexOfUnconfirmed,
        prevWordIndexOfPivot,
      )
    }

  private fun mergeTimestamps(numMergedWords: Int): List<Int> {
    if (numMergedWords == 0) {
      return listOf()
    }
    if (
      prevWordTimestamps.isEmpty() ||
        prevWordIndexOfUnconfirmed == -1 ||
        prevWordIndexOfUnconfirmed == prevWordIndexOfPivot
    ) {
      return currentWordTimestamps
    }
    val numWordsFromPrevWords = prevWordIndexOfPivot - prevWordIndexOfUnconfirmed
    val timestampsFromPrevWords =
      prevWordTimestamps.subList(prevWordIndexOfUnconfirmed, prevWordIndexOfPivot)
    val numWordsFromCurrentWords = numMergedWords - numWordsFromPrevWords
    val timestampsFromCurrentWords =
      currentWordTimestamps.drop(currentWordTimestamps.size - numWordsFromCurrentWords)
    return timestampsFromPrevWords + timestampsFromCurrentWords
  }

  companion object {
    private const val TAG = "ASR"

    @VisibleForTesting
    internal fun levenshtein(s1: String, s2: String): Int {
      val dp = Array(s1.length + 1) { IntArray(s2.length + 1) }
      for (i in 0..s1.length) dp[i][0] = i
      for (j in 0..s2.length) dp[0][j] = j
      for (i in 1..s1.length) {
        for (j in 1..s2.length) {
          val deletionCost = dp[i - 1][j] + 1
          val insertionCost = dp[i][j - 1] + 1
          val matchOrSubstitutionCost = dp[i - 1][j - 1] + if (s1[i - 1] == s2[j - 1]) 0 else 1
          dp[i][j] = minOf(deletionCost, insertionCost, matchOrSubstitutionCost)
        }
      }
      return dp[s1.length][s2.length]
    }

    /**
     * Returns the unconfirmed words which consists of previous words after confirmed point before
     * pivot, plus the current words after the pivot. The pivot point of the current words is
     * calculated based on the previous words after the pivot point for simplicity.
     */
    @VisibleForTesting
    internal fun mergeIntoUnconfirmedText(
      prevWords: List<String>,
      prevWordTimestamps: List<Int>,
      currentWords: List<String>,
      currentWordTimestamps: List<Int>,
      overlapRatio: Float,
      pivotFactor: Float,
      prevWordIndexOfUnconfirmed: Int,
      prevWordIndexOfPivot: Int,
    ): List<String> {
      val useTimestamps = currentWordTimestamps.isNotEmpty() && prevWordTimestamps.isNotEmpty()
      // To make unconfirmed text looks updated incrementally, return only the previous words after
      // the pivot point if unconfirmed text is shorter than the previous words after the pivot.
      val numWordsBeforePivotInCurrent =
        if (useTimestamps) {
          val timestampAtPivot =
            if (prevWordIndexOfPivot >= prevWordTimestamps.size) {
              prevWordTimestamps.last() + 1
            } else {
              prevWordTimestamps[prevWordIndexOfPivot]
            }
          currentWordTimestamps.count { it < timestampAtPivot }
        } else {
          val numWordsOverlapInCurrent =
            ceil(currentWords.size * overlapRatio)
              .toInt()
              .coerceAtLeast(prevWords.size - prevWordIndexOfUnconfirmed)
          (numWordsOverlapInCurrent * pivotFactor).toInt().coerceAtMost(currentWords.size)
        }
      val numWordsAfterPivotInCurrent = currentWords.size - numWordsBeforePivotInCurrent
      if (numWordsAfterPivotInCurrent < prevWords.size - prevWordIndexOfPivot) {
        return prevWords.subList(prevWordIndexOfUnconfirmed, prevWords.size)
      }

      val unconfirmedTextInPrev =
        prevWords.subList(prevWordIndexOfUnconfirmed, prevWordIndexOfPivot)
      val unconfirmedTextInCurrent =
        currentWords.subList(numWordsBeforePivotInCurrent, currentWords.size)
      return unconfirmedTextInPrev + dedupWords(unconfirmedTextInPrev, unconfirmedTextInCurrent)
    }

    /**
     * Removes up to N (=searchWindow) words from the beginning of currentWords if they are
     * duplicated with the end of prevWords.
     */
    @VisibleForTesting
    internal fun dedupWords(
      prevWords: List<String>,
      currentWords: List<String>,
      searchWindow: Int = 2,
    ): List<String> {
      val maxSearchRange = searchWindow.coerceAtMost(minOf(prevWords.size, currentWords.size))
      for (i in maxSearchRange downTo 1) {
        if (
          closeEnough(
            canonicalize(prevWords.subList(prevWords.size - i, prevWords.size)),
            canonicalize(currentWords.subList(0, i)),
          )
        ) {
          return currentWords.subList(i, currentWords.size)
        }
      }
      return currentWords
    }

    @VisibleForTesting
    internal fun canonicalize(words: List<String>): String =
      words.joinToString(" ") { it.lowercase() }.replace(Regex("[\\p{Punct}]"), "")

    @VisibleForTesting
    internal fun closeEnough(l1: List<String>, l2: List<String>): Boolean =
      l1.size == l2.size && l1.zip(l2).all { closeEnough(it.first, it.second) }

    @VisibleForTesting
    internal fun closeEnough(s1: String, s2: String): Boolean {
      val minLen = minOf(s1.length, s2.length)
      val prefix1 = s1.substring(0, minLen)
      val prefix2 = s2.substring(0, minLen)
      return if (minLen <= 3) prefix1 == prefix2 else levenshtein(prefix1, prefix2) <= 1
    }
  }
}
