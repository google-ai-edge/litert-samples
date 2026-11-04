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

package com.google.ai.edge.examples.click_or_code

import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StepTest {
  @Test
  fun readsAClick() {
    val r =
      Step.parse(
        "The sailboat is the third picture.\n</think>\n\nAction: Double-click the sailboat photo.\n\n" +
          "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"double_click\", " +
          "\"coordinate\": [247, 391]}}\n</tool_call>"
      )
    assertEquals("The sailboat is the third picture.", r.thinking)
    assertEquals("Double-click the sailboat photo.", r.actionLine)
    val c = r.first!!
    assertFalse(c.isCode)
    assertEquals("double_click", c.action)
    assertEquals(247.0 to 391.0, c.coordinate)
    assertTrue(c.strictJson)
  }

  @Test
  fun readsACommand() {
    val r =
      Step.parse(
        "Shell is reliable here.</think>\nAction: Rename the files.\n<tool_call>\n{\"name\": \"bash\", " +
          "\"arguments\": {\"command\": \"cd ~/Pictures/trip && for f in *.JPG; do mv \\\"\$f\\\" " +
          "\\\"\${f%.JPG}.jpg\\\"; done\"}}\n</tool_call>"
      )
    val c = r.first!!
    assertTrue(c.isCode)
    assertEquals("cd ~/Pictures/trip && for f in *.JPG; do mv \"\$f\" \"\${f%.JPG}.jpg\"; done", c.command)
  }

  @Test
  fun readsTypographicQuotes() {
    val r =
      Step.parse(
        "x</think>\nAction: Click it.\n<tool_call>\n{“name”: “computer_use”, " +
          "“arguments”: {“action”: “click”, “coordinate”: [5, 6]}}\n</tool_call>"
      )
    assertEquals(5.0 to 6.0, r.first!!.coordinate)
  }

  @Test
  fun readsABrokenCallFieldByField() {
    val r =
      Step.parse(
        "x</think>\nAction: Click it.\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": " +
          "{\"action\": \"click\", \"coordinate\": [12, 34]}\n</tool_call>"
      )
    val c = r.first!!
    assertFalse(c.strictJson)
    assertEquals("click", c.action)
    assertEquals(12.0 to 34.0, c.coordinate)
  }

  @Test
  fun aReplyCutBeforeItsCallHasNoStep() {
    val r = Step.parse("Still thinking about the screen")
    assertEquals("Still thinking about the screen", r.thinking)
    assertNull(r.first)
    assertNull(r.actionLine)
  }

  @Test
  fun aCallCutInsideIsReadAsFarAsItGoes() {
    val r = Step.parse("x</think>\nAction: Run it.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"comm")
    // No field the screen shows is complete: no step to show yet.
    assertNull(r.first?.command)
  }

  @Test
  fun readsEveryCallOfAStep() {
    val r =
      Step.parse(
        "x</think>\nAction: Type and press enter.\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": " +
          "{\"action\": \"type\", \"text\": \"hi\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"computer_use\", " +
          "\"arguments\": {\"action\": \"keypress\", \"keys\": [\"enter\"]}}\n</tool_call>"
      )
    assertEquals(2, r.calls.size)
    assertEquals("hi", r.calls[0].text)
    assertEquals(listOf("enter"), r.calls[1].keys)
  }

  @Test
  fun pointBecomesPixelsAsTheVendorsAgentComputesIt() {
    // round(x / 999 * (w - 1)), halves to even.
    assertEquals(0 to 0, Step.toPixels(0.0 to 0.0, 1920, 1080))
    assertEquals(1919 to 1079, Step.toPixels(999.0 to 999.0, 1920, 1080))
    assertEquals(960 to 540, Step.toPixels(499.5 to 499.5, 1921, 1081))
    assertEquals(1919 to 1079, Step.toPixels(5000.0 to 5000.0, 1920, 1080))
  }

  @Test
  fun requestTurnIsTheAgentsStepTextWithItsScreenshotCollapsed() {
    assertEquals(
      "[This screenshot has been collapsed.]\nPlease generate the next move according to the UI screenshot, " +
        "instruction and previous actions.\n\nInstruction: Open it.\n\nPrevious actions:\nNone",
      Step.requestTurn("  Open it. "),
    )
  }

  @Test
  fun stepTextIsTheAgentsFirstStep() {
    assertEquals(
      "\nPlease generate the next move according to the UI screenshot, instruction and previous " +
        "actions.\n\nInstruction: Open it.\n\nPrevious actions:\nNone",
      Step.stepText("  Open it. "),
    )
  }

  @Test
  fun onlyTheCallForAScreenshotAsksForTheSecondStep() {
    fun reply(call: String) = Step.parse("Reasoning.\n</think>\n\nAction: Do it.\n\n<tool_call>\n$call\n</tool_call>")

    assertTrue(Step.asksForScreenshot(reply("""{"name": "computer_use", "arguments": {"action": "screenshot"}}""")))
    assertFalse(
      Step.asksForScreenshot(reply("""{"name": "computer_use", "arguments": {"action": "click", "coordinate": [1, 2]}}"""))
    )
    assertFalse(Step.asksForScreenshot(reply("""{"name": "bash", "arguments": {"command": "ls"}}""")))
    assertFalse(Step.asksForScreenshot(Step.parse("Reasoning, cut before a call")))
  }

  @Test
  fun bundledSystemPromptIsTheOneThisAppWasCheckedWith() {
    val bytes = javaClass.classLoader!!.getResource("mage_hybrid_system_prompt.txt")!!.readBytes()
    val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    assertNotNull(sha)
    assertEquals(Step.SYSTEM_SHA256, sha)
  }

  @Test
  fun aPromptTooLongIsReadFromEitherMessage() {
    assertTrue(
      Step.isTooLong(
        "Status Code: 3. Message: Input token ids are too long. " +
          "Exceeding the maximum number of tokens allowed: 4375 >= 4096"
      )
    )
    assertTrue(Step.isTooLong("Prefill input length exceeds the file's signatures"))
    assertTrue(!Step.isTooLong("Status Code: 13. Message: something else"))
    assertTrue(!Step.isTooLong(null))
  }

  @Test
  fun examplesAreBundled() {
    val json = javaClass.classLoader!!.getResource("examples/examples.json")!!.readText()
    val examples = Examples.parse(json)
    assertTrue(examples.isNotEmpty())
    for (e in examples) {
      assertNotNull(e.file, javaClass.classLoader!!.getResource("examples/${e.file}"))
      assertTrue(e.id, e.requests.isNotEmpty())
    }
  }
}
