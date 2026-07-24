package com.example.mobilenetlitert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassificationPostprocessorTest {
  @Test
  fun topK_sortsScoresAndMapsLabels() {
    val logits = floatArrayOf(0.1f, 5.0f, 2.0f, 7.0f)
    val labels = listOf("zero", "one", "two", "three")

    val top = ClassificationPostprocessor.topK(logits, labels, 3)

    assertEquals(listOf("three", "one", "two"), top.map { it.label })
    assertEquals(listOf(3, 1, 2), top.map { it.index })
    assertTrue(top.zipWithNext().all { (a, b) -> a.score >= b.score })
  }
}
