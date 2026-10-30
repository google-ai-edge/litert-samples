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

import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.time.temporal.TemporalAdjusters
import org.json.JSONArray
import org.json.JSONObject

data class Contact(val name: String)

data class Event(
  val id: Long,
  val title: String,
  val start: LocalDateTime,
  val end: LocalDateTime,
  val location: String? = null,
  val attendees: List<String> = emptyList(),
)

data class Draft(val id: Long, val recipient: String, val message: String)

/**
 * The app's own calendar, contacts and outbox. Nothing here touches the phone's calendar,
 * contacts or messages: the tools the model calls change only this state, which the app keeps in
 * one JSON file in its private storage.
 */
class PlannerState(
  val contacts: MutableList<Contact> = mutableListOf(),
  val events: MutableList<Event> = mutableListOf(),
  val drafts: MutableList<Draft> = mutableListOf(),
) {
  private var nextId = 1L

  fun newId(): Long = nextId++

  fun sortedEvents(): List<Event> = events.sortedBy { it.start }

  fun addContact(name: String): Boolean {
    val clean = name.trim().replace(Regex("\\s+"), " ")
    if (clean.isEmpty() || contacts.any { norm(it.name) == norm(clean) }) return false
    contacts += Contact(clean)
    return true
  }

  fun addEvent(title: String, start: LocalDateTime, minutes: Long = 60): Event {
    val event = Event(newId(), title.trim(), start, start.plusMinutes(minutes))
    events += event
    return event
  }

  /** The contact a name from the model means: the full name, or a first or last name that only
   * one contact has (case and punctuation ignored). Null when no contact or more than one fits. */
  fun resolveContact(name: String): Contact? {
    val q = norm(name)
    if (q.isEmpty()) return null
    contacts.firstOrNull { norm(it.name) == q }?.let { return it }
    val qWords = q.split(" ")
    val byPart = contacts.filter { c -> norm(c.name).split(" ").let { parts -> qWords.all { it in parts } } }
    return byPart.singleOrNull()
  }

  /** The events a title or keyword from the model can mean, best match first. A word counts when
   * it is not a filler word or a time; the events with the most matching words win. [day]
   * narrows the search to one date. */
  fun matchEvents(query: String, day: LocalDate? = null): List<Event> {
    val pool = events.filter { day == null || it.start.toLocalDate() == day }
    val q = norm(query)
    pool.filter { norm(it.title) == q }.let { if (it.isNotEmpty()) return it }
    val words = q.split(" ").filter { it.isNotEmpty() && it !in FILLER && !it.any(Char::isDigit) }
    if (words.isEmpty()) return emptyList()
    val scored = pool.map { e -> e to norm(e.title).split(" ").count { it in words } }.filter { it.second > 0 }
    val best = scored.maxOfOrNull { it.second } ?: return emptyList()
    return scored.filter { it.second == best }.map { it.first }
  }

  fun toJson(): JSONObject =
    JSONObject()
      .put("contacts", JSONArray(contacts.map { it.name }))
      .put(
        "events",
        JSONArray(
          events.map { e ->
            JSONObject()
              .put("title", e.title)
              .put("start", e.start.format(ISO_MINUTE))
              .put("end", e.end.format(ISO_MINUTE))
              .put("location", e.location ?: JSONObject.NULL)
              .put("attendees", JSONArray(e.attendees))
          }
        ),
      )
      .put(
        "drafts",
        JSONArray(drafts.map { JSONObject().put("recipient", it.recipient).put("message", it.message) }),
      )

  fun save(file: File) {
    val tmp = File(file.parentFile, file.name + ".tmp")
    tmp.writeText(toJson().toString(1))
    tmp.renameTo(file)
  }

  companion object {
    val ISO_MINUTE: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm")

    private val FILLER =
      setOf("my", "the", "a", "an", "with", "to", "at", "on", "in", "for", "of", "and", "appointment",
        "event", "tonight", "today", "tomorrow", "pm", "am")

    /** Lower case, accents and punctuation removed, single spaces. */
    fun norm(s: String): String {
      val lower = java.text.Normalizer.normalize(s.lowercase(), java.text.Normalizer.Form.NFD)
        .replace(Regex("\\p{Mn}+"), "")
      return lower.replace(Regex("[^\\p{L}\\p{N}]+"), " ").trim()
    }

    /** The example state, relative to [now]: six fictional contacts and the week's events. */
    fun seed(now: LocalDateTime): PlannerState {
      val s = PlannerState()
      listOf("Ken Walker", "Maya Patel", "Sam Rivera", "Priya Shah", "Leo Martin", "Dana Brooks")
        .forEach { s.addContact(it) }
      val today = now.toLocalDate()
      fun next(day: java.time.DayOfWeek) = today.with(TemporalAdjusters.next(day))
      s.addEvent("Airport taxi", today.atTime(18, 0), 30)
      s.addEvent("Dinner with Ken", today.atTime(19, 0), 90)
      s.addEvent("Dentist", next(java.time.DayOfWeek.MONDAY).atTime(9, 0))
      s.addEvent("Coffee with Sam", next(java.time.DayOfWeek.TUESDAY).atTime(11, 0))
      s.addEvent("Lunch with Priya", next(java.time.DayOfWeek.WEDNESDAY).atTime(12, 30))
      s.addEvent("Book club", next(java.time.DayOfWeek.THURSDAY).atTime(19, 0), 120)
      return s
    }

    fun load(file: File): PlannerState? =
      runCatching {
          val o = JSONObject(file.readText())
          val s = PlannerState()
          o.getJSONArray("contacts").let { a -> for (i in 0 until a.length()) s.addContact(a.getString(i)) }
          o.getJSONArray("events").let { a ->
            for (i in 0 until a.length()) {
              val e = a.getJSONObject(i)
              val att = e.optJSONArray("attendees")
              s.events +=
                Event(
                  s.newId(),
                  e.getString("title"),
                  LocalDateTime.parse(e.getString("start")),
                  LocalDateTime.parse(e.getString("end")),
                  if (e.isNull("location")) null else e.optString("location"),
                  if (att == null) emptyList() else (0 until att.length()).map { att.getString(it) },
                )
            }
          }
          o.getJSONArray("drafts").let { a ->
            for (i in 0 until a.length()) {
              val d = a.getJSONObject(i)
              s.drafts += Draft(s.newId(), d.getString("recipient"), d.getString("message"))
            }
          }
          s
        }
        .getOrNull()
  }
}
