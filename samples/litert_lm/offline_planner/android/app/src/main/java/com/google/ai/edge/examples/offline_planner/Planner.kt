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

import com.google.gson.JsonArray
import com.google.gson.JsonParser
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/** What the model sees: the tools (JSON schemas, rendered by the chat template into the system
 * turn) and the system instruction (the phone's date and time and a few rules). */
object Planner {
  val TOOL_NAMES = listOf("create_event", "move_event", "delete_event", "draft_message")

  /** The four tools, in the JSON schema form the model's chat template renders. */
  val TOOLS_JSON =
    """
    [
      {"type": "function", "function": {"name": "create_event",
        "description": "Adds an event to the calendar.",
        "parameters": {"type": "object", "properties": {
          "title": {"type": "string"},
          "start": {"type": "string", "description": "Start, as YYYY-MM-DDTHH:MM:SS."},
          "end": {"type": "string", "description": "End, as YYYY-MM-DDTHH:MM:SS. Only if the user gives one."},
          "location": {"type": "string"},
          "attendees": {"type": "array", "items": {"type": "string"}, "description": "Names of the contacts who take part."}},
          "required": ["title", "start"]}}},
      {"type": "function", "function": {"name": "move_event",
        "description": "Changes the start time of an event that is already in the calendar.",
        "parameters": {"type": "object", "properties": {
          "title": {"type": "string", "description": "The event's title, or a word from it."},
          "new_start": {"type": "string", "description": "The new start the user asks for, as YYYY-MM-DDTHH:MM:SS."}},
          "required": ["title", "new_start"]}}},
      {"type": "function", "function": {"name": "delete_event",
        "description": "Removes an event from the calendar.",
        "parameters": {"type": "object", "properties": {
          "title": {"type": "string", "description": "The event's title, or a word from it."},
          "date": {"type": "string", "description": "The event's day, as YYYY-MM-DD."}},
          "required": ["title"]}}},
      {"type": "function", "function": {"name": "draft_message",
        "description": "Writes a text message to a contact and saves it in the outbox (it is not sent). Use it when the user asks to tell, message or let someone know.",
        "parameters": {"type": "object", "properties": {
          "recipient": {"type": "string", "description": "The contact's name."},
          "message": {"type": "string", "description": "The text of the message."}},
          "required": ["recipient", "message"]}}}
    ]
    """
      .trimIndent()

  /** The tools as one JSON array for `ConversationConfig.extraContext["tools"]`; a Gson tree
   * keeps the key order of the text above. */
  fun toolsJson(): JsonArray = JsonParser.parseString(TOOLS_JSON).asJsonArray

  private val DAY_FORMAT = DateTimeFormatter.ofPattern("EEEE, yyyy-MM-dd HH:mm", Locale.US)

  private val WEEK_DAY = DateTimeFormatter.ofPattern("EEEE yyyy-MM-dd", Locale.US)
  private val EVENT_TIME = DateTimeFormatter.ofPattern("EEEE yyyy-MM-dd HH:mm", Locale.US)
  private val CLOCK = DateTimeFormatter.ofPattern("HH:mm", Locale.US)

  /** The system instruction; the app fills in the braces for each request. */
  val SYSTEM_TEMPLATE =
    """
    You are the assistant inside a calendar and messages app on the user's phone. You can call the app's tools.
    Now: {now} ({zone}).
    The next seven days: {week}.
    Contacts: {contacts}.
    Calendar:
    {calendar}

    Rules:
    - Call every tool the request needs, in the order the user asks for, with arguments from the request, the dates and the calendar above. Leave out optional arguments the user does not give, and never make up values.
    - Write all the calls in this one reply, one after another. No tool result comes back before the user approves them.
    - When the user asks you to tell, message or let someone know, also call draft_message for each person.
    - If a detail the tool needs is missing, ask one short question instead of calling a tool.
    - Answer questions that need no tool directly, without calling a tool.
    - If no tool can do what is asked, say so in one sentence.
    """
      .trimIndent()

  /** The next seven days with their dates, so the model looks up "on Tuesday" instead of
   * counting. */
  fun week(now: LocalDateTime): String =
    (1..7).joinToString(", ") { i ->
      now.toLocalDate().plusDays(i.toLong()).format(WEEK_DAY) + if (i == 1) " (tomorrow)" else ""
    }

  /** The calendar from today on (at most [MAX_EVENTS] events), one line each. */
  fun calendar(now: LocalDateTime, state: PlannerState): String {
    val events = state.sortedEvents().filter { !it.start.toLocalDate().isBefore(now.toLocalDate()) }.take(MAX_EVENTS)
    if (events.isEmpty()) return "- (empty)"
    return events.joinToString("\n") { "- ${it.title}: ${it.start.format(EVENT_TIME)}-${it.end.format(CLOCK)}" }
  }

  private const val MAX_EVENTS = 12

  /** The system instruction for one request at [now] in [zone], with the app's contacts and
   * calendar. */
  fun systemInstruction(now: LocalDateTime, zone: ZoneId, state: PlannerState): String =
    SYSTEM_TEMPLATE.replace("{now}", now.format(DAY_FORMAT))
      .replace("{zone}", zone.id)
      .replace("{week}", week(now))
      .replace("{contacts}", state.contacts.joinToString(", ") { it.name }.ifEmpty { "(none)" })
      .replace("{calendar}", calendar(now, state))

  /** The examples in the app's request list. */
  val EXAMPLES =
    listOf(
      "My flight lands two hours late. Move dinner with Ken to 9 pm, let him know, and delete the 6 pm taxi.",
      "Put lunch with Ken tomorrow at 12:30 on my calendar and let him know.",
      "Schedule a project meeting on Monday at 3 pm and tell Maya and Leo about it.",
      "Add a call with Dana on Tuesday at half past nine in the morning.",
      "How many minutes are in two and a half hours?",
    )
}
