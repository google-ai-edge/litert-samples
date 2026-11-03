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

import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import org.json.JSONArray

/** One call as the model wrote it: the tool's name and its arguments, in order. */
data class ToolCall(val name: String, val args: Map<String, String>)

/**
 * What a reply holds: the calls, the text outside them (the answer when there is no call), and a
 * problem when the reply cannot be read as calls.
 */
data class ParsedReply(val calls: List<ToolCall>, val answer: String, val problem: String?)

/**
 * LFM3 writes each call in its XML form:
 * ```
 * <tool_call>
 * <function=create_event>
 * <parameter=title>
 * Lunch with Ken
 * </parameter>
 * ...
 * </function>
 * </tool_call>
 * ```
 * LiteRT-LM returns it as plain text for this model (the bundle declares the generic model type),
 * so the app parses it here.
 */
object ReplyParser {
  private val THINK = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)
  private val CALL = Regex("<tool_call>(.*?)</tool_call>", RegexOption.DOT_MATCHES_ALL)
  private val FUNCTION = Regex("<function=([^>\\n]+)>(.*?)</function>", RegexOption.DOT_MATCHES_ALL)
  private val PARAM = Regex("<parameter=([^>\\n]+)>\\n?(.*?)\\n?</parameter>", RegexOption.DOT_MATCHES_ALL)

  /** Parses [text]. With [complete] false (the reply is still streaming) an unclosed call is not
   * an error yet. */
  fun parse(text: String, complete: Boolean = true): ParsedReply {
    // An empty thought block, or a stray end-of-thought tag, can come before the answer.
    val clean = THINK.replace(text, "").replace("</think>", "").replace("<think>", "")
    val calls = mutableListOf<ToolCall>()
    var problem: String? = null
    for (m in CALL.findAll(clean)) {
      val f = FUNCTION.find(m.groupValues[1])
      if (f == null) {
        problem = "A tool call without a function: ${m.value.take(120)}"
        continue
      }
      val args = LinkedHashMap<String, String>()
      for (p in PARAM.findAll(f.groupValues[2])) {
        args[p.groupValues[1].trim()] = p.groupValues[2].trim()
      }
      calls += ToolCall(f.groupValues[1].trim(), args)
    }
    var rest = CALL.replace(clean, "")
    if ("<tool_call>" in rest) {
      if (complete) problem = "The reply ends inside a tool call."
      rest = rest.substringBefore("<tool_call>")
    }
    if (problem == null && complete && ("<function=" in rest || "</tool_call>" in rest)) {
      problem = "A tool call is not in the expected form."
    }
    return ParsedReply(calls, rest.trim(), problem)
  }
}

/** A call checked against the app's tools and state: what it would do, or why it will not run. */
class Action(val call: ToolCall, val summary: String, val refusal: String?, private val apply: (() -> Change)?) {
  val ok: Boolean
    get() = refusal == null

  fun run(): Change = checkNotNull(apply) { "refused: $refusal" }.invoke()
}

/** What a call changed: a line for the screen and the event or draft it touched. */
data class Change(val line: String, val eventId: Long? = null, val draftId: Long? = null)

object Actions {
  private val SHOW = DateTimeFormatter.ofPattern("EEE d MMM, HH:mm", Locale.US)

  fun show(t: LocalDateTime): String = t.format(SHOW)

  /** Reads `YYYY-MM-DDTHH:MM[:SS]` (a space instead of the `T` too). */
  fun parseDateTime(s: String): LocalDateTime? {
    val m = Regex("^(\\d{4}-\\d{2}-\\d{2})[T ](\\d{1,2}):(\\d{2})(?::(\\d{2}))?$").find(s.trim()) ?: return null
    return runCatching {
        LocalDate.parse(m.groupValues[1])
          .atTime(m.groupValues[2].toInt(), m.groupValues[3].toInt(), m.groupValues[4].ifEmpty { "0" }.toInt())
      }
      .getOrNull()
  }

  /** A list argument: a JSON array, or names separated by commas. */
  private fun names(v: String): List<String> =
    runCatching { JSONArray(v).let { a -> (0 until a.length()).map { a.getString(it) } } }
      .getOrElse { v.split(",") }
      .map { it.trim() }
      .filter { it.isNotEmpty() }

  /** Checks every call of a reply. Nothing runs here; [Action.run] does it after the user taps
   * Run. */
  fun check(calls: List<ToolCall>, state: PlannerState): List<Action> = calls.map { check(it, state) }

  private fun refuse(call: ToolCall, summary: String, why: String) = Action(call, summary, why, null)

  fun check(call: ToolCall, s: PlannerState): Action {
    val a = call.args
    fun missing(vararg keys: String) = keys.firstOrNull { a[it].isNullOrBlank() }
    return when (call.name) {
      "create_event" -> {
        val title = a["title"].orEmpty()
        val summary = "Add \"$title\""
        missing("title", "start")?.let { return refuse(call, summary, "No $it given.") }
        val start = parseDateTime(a.getValue("start")) ?: return refuse(call, summary, "Cannot read the start \"${a["start"]}\".")
        val end = a["end"]?.let { parseDateTime(it) }?.takeIf { it.isAfter(start) } ?: start.plusMinutes(60)
        val people = a["attendees"]?.let { names(it) }.orEmpty()
        val unknown = people.filter { s.resolveContact(it) == null }
        val line = "$summary on ${show(start)}"
        if (unknown.isNotEmpty()) return refuse(call, line, "Not in your contacts: ${unknown.joinToString()}.")
        val attendees = people.mapNotNull { s.resolveContact(it)?.name }
        Action(call, line + if (attendees.isEmpty()) "" else " with ${attendees.joinToString()}", null) {
          val e = Event(s.newId(), title, start, end, a["location"]?.ifBlank { null }, attendees)
          s.events += e
          Change("Added \"${e.title}\" on ${show(e.start)}", eventId = e.id)
        }
      }
      "move_event" -> {
        val title = a["title"].orEmpty()
        val summary = "Move \"$title\""
        missing("title", "new_start")?.let { return refuse(call, summary, "No $it given.") }
        val start = parseDateTime(a.getValue("new_start")) ?: return refuse(call, summary, "Cannot read the time \"${a["new_start"]}\".")
        val matches = s.matchEvents(title)
        val e = matches.singleOrNull() ?: return refuse(call, "$summary to ${show(start)}", noneOrMany(title, matches.size))
        Action(call, "Move \"${e.title}\" from ${show(e.start)} to ${show(start)}", null) {
          val i = s.events.indexOfFirst { it.id == e.id }
          if (i < 0) {
            Change("\"${e.title}\" is no longer in the calendar; not moved")
          } else {
            s.events[i] = e.copy(start = start, end = start.plus(java.time.Duration.between(e.start, e.end)))
            Change("Moved \"${e.title}\" to ${show(start)}", eventId = e.id)
          }
        }
      }
      "delete_event" -> {
        val title = a["title"].orEmpty()
        val summary = "Delete \"$title\""
        missing("title")?.let { return refuse(call, summary, "No $it given.") }
        val day = a["date"]?.let { runCatching { LocalDate.parse(it.trim().take(10)) }.getOrNull() }
        val matches = s.matchEvents(title, day)
        val e = matches.singleOrNull() ?: return refuse(call, summary, noneOrMany(title, matches.size))
        Action(call, "Delete \"${e.title}\" (${show(e.start)})", null) {
          s.events.removeAll { it.id == e.id }
          Change("Deleted \"${e.title}\" (${show(e.start)})")
        }
      }
      "draft_message" -> {
        val who = a["recipient"].orEmpty()
        val summary = "Draft a message to $who"
        missing("recipient", "message")?.let { return refuse(call, summary, "No $it given.") }
        val contact = s.resolveContact(who) ?: return refuse(call, summary, "Not in your contacts: $who.")
        val text = a.getValue("message")
        Action(call, "Draft to ${contact.name}: \"$text\"", null) {
          val d = Draft(s.newId(), contact.name, text)
          s.drafts += d
          Change("Saved a draft to ${contact.name} in the outbox (not sent)", draftId = d.id)
        }
      }
      else ->
        refuse(call, "Call ${call.name}", "This app has no tool called \"${call.name}\" (it has ${Planner.TOOL_NAMES.joinToString()}).")
    }
  }

  private fun noneOrMany(title: String, n: Int) =
    if (n == 0) "No event in the calendar matches \"$title\"." else "$n events match \"$title\"; nothing changed."
}
