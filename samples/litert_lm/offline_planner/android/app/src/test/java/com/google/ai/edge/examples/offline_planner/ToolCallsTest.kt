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

package com.google.ai.edge.examples.offline_planner

import java.time.LocalDateTime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolCallsTest {
  private val now = LocalDateTime.of(2026, 10, 30, 9, 0)

  private fun call(name: String, vararg args: Pair<String, String>) =
    "<tool_call>\n<function=$name>\n" +
      args.joinToString("") { "<parameter=${it.first}>\n${it.second}\n</parameter>\n" } +
      "</function>\n</tool_call>"

  @Test
  fun parsesTwoCallsInOrder() {
    val text =
      call("create_event", "title" to "Lunch with Ken", "start" to "2026-10-31T12:30:00") + "\n\n" +
        call("draft_message", "recipient" to "Ken Walker", "message" to "Lunch tomorrow at 12:30?")
    val r = ReplyParser.parse(text)
    assertNull(r.problem)
    assertEquals(listOf("create_event", "draft_message"), r.calls.map { it.name })
    assertEquals("2026-10-31T12:30:00", r.calls[0].args["start"])
    assertEquals("Lunch tomorrow at 12:30?", r.calls[1].args["message"])
    assertEquals("", r.answer)
  }

  @Test
  fun dropsThoughtTags() {
    val r = ReplyParser.parse("<think></think>17 + 25 = 42.</think>42")
    assertTrue(r.calls.isEmpty())
    assertEquals("17 + 25 = 42.42", r.answer)
  }

  @Test
  fun aCutOffCallIsAProblemOnlyWhenTheReplyIsComplete() {
    val text = "<tool_call>\n<function=create_event>\n<parameter=title>\nLunch"
    assertNull(ReplyParser.parse(text, complete = false).problem)
    assertEquals("The reply ends inside a tool call.", ReplyParser.parse(text).problem)
  }

  @Test
  fun refusesAnUnknownToolAndAnUnknownPerson() {
    val s = PlannerState.seed(now)
    val alarm = Actions.check(ToolCall("set_alarm", mapOf("time" to "06:30")), s)
    assertTrue(alarm.refusal!!.startsWith("This app has no tool called \"set_alarm\""))
    val bob = Actions.check(ToolCall("draft_message", mapOf("recipient" to "Bob", "message" to "Hi")), s)
    assertEquals("Not in your contacts: Bob.", bob.refusal)
  }

  @Test
  fun resolvesFirstNamesAndEventWords() {
    val s = PlannerState.seed(now)
    assertEquals("Ken Walker", s.resolveContact("ken")?.name)
    assertEquals("Dentist", s.matchEvents("my dentist appointment").single().title)
    assertEquals("Airport taxi", s.matchEvents("6 pm taxi").single().title)
    assertTrue(s.matchEvents("haircut").isEmpty())
  }

  @Test
  fun runsTheCheckedActions() {
    val s = PlannerState.seed(now)
    val actions =
      Actions.check(
        listOf(
          ToolCall("move_event", mapOf("title" to "Dinner with Ken", "new_start" to "2026-10-30T21:30:00")),
          ToolCall("draft_message", mapOf("recipient" to "Ken", "message" to "Dinner at 9:30?")),
          ToolCall("delete_event", mapOf("title" to "taxi", "date" to "2026-10-30")),
        ),
        s,
      )
    assertTrue(actions.all { it.ok })
    actions.forEach { it.run() }
    val dinner = s.events.single { it.title == "Dinner with Ken" }
    assertEquals(LocalDateTime.of(2026, 10, 30, 21, 30), dinner.start)
    assertNull(s.events.firstOrNull { it.title == "Airport taxi" })
    assertEquals("Ken Walker", s.drafts.single().recipient)
  }

  @Test
  fun readsTimesWithAndWithoutSeconds() {
    assertEquals(LocalDateTime.of(2026, 11, 2, 15, 0), Actions.parseDateTime("2026-11-02T15:00"))
    assertEquals(LocalDateTime.of(2026, 11, 2, 9, 30), Actions.parseDateTime("2026-11-02 9:30:00"))
    assertNull(Actions.parseDateTime("Monday 3 pm"))
    assertNotNull(Actions.parseDateTime(" 2026-11-02T15:00:00 "))
  }
}
