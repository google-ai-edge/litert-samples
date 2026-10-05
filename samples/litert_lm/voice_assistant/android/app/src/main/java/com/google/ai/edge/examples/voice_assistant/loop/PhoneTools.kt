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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// voice/src/main/kotlin/io/github/johnrocky/hfmodels/voice/PhoneTools.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import android.app.AlarmManager
import android.content.ActivityNotFoundException
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.provider.AlarmClock
import android.provider.CalendarContract
import java.text.ParsePosition
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Phone tools that do the real thing on this phone: the alarm and the timer land in the Clock app
 * (`AlarmClock` intents with `EXTRA_SKIP_UI`), events in the app's own local "Phone Agent" calendar
 * (created the first time the app reads or writes it, which is right after the first load, when
 * the screen shows tomorrow's events; account calendars are never read or written). A tool that
 * cannot do its job says so in its result. The app declares what they need:
 * `com.android.alarm.permission.SET_ALARM`, `android.permission.READ_CALENDAR` and
 * `WRITE_CALENDAR` (granted at run time).
 */
object PhoneTools {
  const val CALENDAR_NAME = "Phone Agent"

  /** get_current_datetime, get_calendar_events, set_alarm, add_calendar_event, set_timer. */
  fun all(context: Context): List<VoiceTool> {
    val app = context.applicationContext
    val calendar = CalendarTool(app)
    return listOf(ClockTool(), calendar.read, AlarmTool(app), calendar.add, TimerTool(app))
  }

  /**
   * What Android itself reports: the next alarm the OS will fire, and the "Phone Agent" calendar
   * for tomorrow (for a screen after a run).
   */
  suspend fun phoneState(context: Context): String =
    withContext(Dispatchers.IO) {
      val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
      val next =
        am.nextAlarmClock?.let {
          SimpleDateFormat("EEE HH:mm", Locale.US).format(Date(it.triggerTime))
        } ?: "none"
      val day = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date(tomorrowNoon()))
      val events =
        try {
          val raw = CalendarTool(context.applicationContext).events(day)
          if (raw.startsWith("No events")) {
            "none"
          } else {
            val a = JSONArray(raw)
            (0 until a.length()).joinToString("\n") { i ->
              val o = a.getJSONObject(i)
              val start = o.getString("start").takeLast(5)
              val end = o.getString("end").takeLast(5)
              "   $start–$end  ${o.getString("title")}"
            }
          }
        } catch (e: Exception) {
          "?"
        }
      "Next alarm (Android): $next\nCalendar, $day:\n$events"
    }

  internal fun tomorrowNoon(): Long =
    Calendar.getInstance()
      .apply {
        add(Calendar.DAY_OF_YEAR, 1)
        set(Calendar.HOUR_OF_DAY, 12)
        set(Calendar.MINUTE, 0)
      }
      .timeInMillis
}

/** `get_current_datetime`: the local date and time with the day of the week. */
class ClockTool : VoiceTool {
  override val name = "get_current_datetime"
  override val description =
    "Returns the current local date and time, including the day of the week."
  override val parameters = emptyList<ToolParam>()

  override suspend fun call(args: Map<String, Any?>): String =
    SimpleDateFormat("EEEE, yyyy-MM-dd HH:mm", Locale.US).format(Date())
}

/**
 * `set_alarm(hour, minute, label)`: an alarm in the Clock app, without showing its UI, confirmed
 * by Android: after the intent the tool waits up to 1.5 s for [AlarmManager.getNextAlarmClock] to
 * report the requested time (its next occurrence, today or tomorrow). Android 10 and later drop an
 * activity start from an app without a visible activity without an exception (a locked Galaxy
 * S26: BAL_BLOCK, result code 102); the result is then an `Error:` that says so. Android reports
 * one next alarm: when an alarm at the same minute or an earlier one is already next, it cannot
 * show this one, and the result says that it was requested and what Android reports instead
 * ([AlarmCheck.unconfirmed]), neither success nor an error.
 */
class AlarmTool(private val context: Context) : VoiceTool {
  override val name = "set_alarm"
  override val description = "Sets an alarm on this phone."
  override val isAction = true
  override val parameters =
    listOf(
      ToolParam("hour", "integer", "Hour in 24-hour time (0-23)."),
      ToolParam("minute", "integer", "Minute (0-59)."),
      ToolParam("label", "string", "Short label shown with the alarm."),
    )

  override suspend fun call(args: Map<String, Any?>): String {
    val hour = ToolArgs.int(args, "hour")
    val minute = ToolArgs.int(args, "minute")
    val label = ToolArgs.stringOrNull(args, "label") ?: ""
    require(hour in 0..23 && minute in 0..59) { "hour must be 0-23 and minute 0-59" }
    val i =
      Intent(AlarmClock.ACTION_SET_ALARM)
        .putExtra(AlarmClock.EXTRA_HOUR, hour)
        .putExtra(AlarmClock.EXTRA_MINUTES, minute)
        .putExtra(AlarmClock.EXTRA_MESSAGE, label)
        .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val expected = AlarmCheck.nextOccurrence(System.currentTimeMillis(), hour, minute)
    val before = am.nextAlarmClock?.triggerTime
    try {
      context.startActivity(i)
    } catch (e: ActivityNotFoundException) {
      return "Error: no clock app can set alarms on this phone"
    }
    val unconfirmed = AlarmCheck.unconfirmed(before, expected, hour, minute, label)
    if (unconfirmed != null) {
      return unconfirmed
    }
    val deadline = System.nanoTime() + AlarmCheck.CONFIRM_MS * 1_000_000
    while (true) {
      if (AlarmCheck.matches(am.nextAlarmClock?.triggerTime, expected)) {
        return "Alarm set for ${AlarmCheck.requested(hour, minute, label)}"
      }
      if (System.nanoTime() >= deadline) {
        return "Error: the Clock app did not take the alarm " +
          "(the screen must be on and this app visible)"
      }
      delay(AlarmCheck.POLL_MS)
    }
  }
}

/** [AlarmTool]'s reading of Android's next alarm clock (local time). */
internal object AlarmCheck {
  const val CONFIRM_MS = 1_500L
  const val POLL_MS = 100L

  /**
   * The next moment after [now] at which the clock reads [hour]:[minute]: today, or tomorrow when
   * it has passed.
   */
  fun nextOccurrence(
    now: Long,
    hour: Int,
    minute: Int,
    zone: TimeZone = TimeZone.getDefault(),
  ): Long {
    val c =
      Calendar.getInstance(zone).apply {
        timeInMillis = now
        set(Calendar.HOUR_OF_DAY, hour)
        set(Calendar.MINUTE, minute)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
      }
    if (c.timeInMillis <= now) {
      c.add(Calendar.DAY_OF_YEAR, 1)
    }
    return c.timeInMillis
  }

  /** Android's next alarm clock is the requested alarm (within its minute). */
  fun matches(next: Long?, expected: Long): Boolean =
    next != null && next >= expected && next < expected + 60_000

  /** An alarm before the requested one was already next: Android keeps reporting that one. */
  fun hidden(before: Long?, expected: Long): Boolean = before != null && before < expected

  /** "07:30 (Wake Up)". */
  fun requested(hour: Int, minute: Int, label: String): String =
    String.format(Locale.US, "%02d:%02d (%s)", hour, minute, label)

  /**
   * The result when Android's next alarm before the request ([before]) already hides it: one at
   * the same minute ([matches]) or an earlier one ([hidden]). Android keeps reporting that alarm,
   * so the request cannot be confirmed; the text says what Android reports. Null when nothing
   * hides the request.
   */
  fun unconfirmed(
    before: Long?,
    expected: Long,
    hour: Int,
    minute: Int,
    label: String,
    zone: TimeZone = TimeZone.getDefault(),
  ): String? {
    if (before == null || !(matches(before, expected) || hidden(before, expected))) {
      return null
    }
    val format = SimpleDateFormat("EEE HH:mm", Locale.US).apply { timeZone = zone }
    val next = format.format(Date(before))
    val already = if (matches(before, expected)) "was already" else "is"
    return "Alarm requested for ${requested(hour, minute, label)}. Android's next alarm " +
      "$already $next, so this one could not be confirmed."
  }
}

/**
 * `set_timer(minutes, label)`: a countdown in the Clock app, without showing its UI. Android has
 * no public API to read the Clock app's timers, so the tool cannot confirm one and its result says
 * the timer was requested. Call it with the screen on and the app visible: Android 10 and later
 * drop an activity start from an app without a visible activity without an exception.
 */
class TimerTool(private val context: Context) : VoiceTool {
  override val name = "set_timer"
  override val description = "Starts a countdown timer on this phone."
  override val isAction = true
  override val parameters =
    listOf(
      ToolParam("minutes", "integer", "Length of the timer in minutes."),
      ToolParam("label", "string", "Short label shown with the timer."),
    )

  override suspend fun call(args: Map<String, Any?>): String {
    val minutes = ToolArgs.int(args, "minutes")
    val label = ToolArgs.stringOrNull(args, "label") ?: ""
    require(minutes in 1..1440) { "minutes must be 1-1440" }
    val i =
      Intent(AlarmClock.ACTION_SET_TIMER)
        .putExtra(AlarmClock.EXTRA_LENGTH, minutes * 60)
        .putExtra(AlarmClock.EXTRA_MESSAGE, label)
        .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    return try {
      context.startActivity(i)
      "Timer requested: $minutes min ($label)"
    } catch (e: ActivityNotFoundException) {
      "Error: no clock app can start timers on this phone"
    }
  }
}

/** The seconds of "10:00:00" (group 1 keeps the minutes). */
private val SECONDS = Regex("(:\\d{2}):\\d{2}(\\.\\d+)?$")

/**
 * `get_calendar_events(date)` ([read]) and `add_calendar_event(title, start, end, location)`
 * ([add]) on one calendar: the app's own local "Phone Agent" calendar, created the first time
 * the app reads or writes it (right after the first load, for the screen's phone state). An
 * account calendar is never touched, so a run cannot leak or alter someone's real schedule.
 */
class CalendarTool(private val context: Context) {
  // Strict: a lenient format reads "73:00" as 01:00 three days later and "2026-02-30" as March 2.
  private val fmtMin = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply { isLenient = false }

  val read: VoiceTool =
    object : VoiceTool {
      override val name = "get_calendar_events"
      override val description = "Lists the events already on the phone calendar for one day."
      override val parameters =
        listOf(ToolParam("date", "string", "The day to list, as YYYY-MM-DD."))

      override suspend fun call(args: Map<String, Any?>): String =
        withContext(Dispatchers.IO) { granted { events(ToolArgs.string(args, "date")) } }
    }

  val add: VoiceTool =
    object : VoiceTool {
      override val name = "add_calendar_event"
      override val description = "Adds an event to the phone calendar."
      override val isAction = true
      override val parameters =
        listOf(
          ToolParam("title", "string", "Event title."),
          ToolParam("start", "string", "Start time as YYYY-MM-DD HH:MM."),
          ToolParam("end", "string", "End time as YYYY-MM-DD HH:MM."),
          ToolParam("location", "string", "Where the event takes place.", required = false),
        )

      override suspend fun call(args: Map<String, Any?>): String =
        withContext(Dispatchers.IO) {
          granted {
            addEvent(
              ToolArgs.string(args, "title"),
              ToolArgs.string(args, "start"),
              ToolArgs.string(args, "end"),
              ToolArgs.stringOrNull(args, "location") ?: "",
            )
          }
        }
    }

  /**
   * [block]'s result; without the calendar permission one plain sentence instead of Android's
   * text, which names the provider and the permission with dots and would be said in pieces.
   */
  internal fun granted(block: () -> String): String =
    try {
      block()
    } catch (e: SecurityException) {
      "Error: the calendar permission is not granted"
    }

  internal fun events(date: String): String {
    val (begin, end) =
      dayWindow(date) ?: throw IllegalArgumentException("bad date '$date' (use YYYY-MM-DD)")
    val uri =
      CalendarContract.Instances.CONTENT_URI.buildUpon()
        .appendPath(begin.toString())
        .appendPath(end.toString())
        .build()
    val proj =
      arrayOf(
        CalendarContract.Instances.TITLE,
        CalendarContract.Instances.BEGIN,
        CalendarContract.Instances.END,
        CalendarContract.Instances.EVENT_LOCATION,
      )
    val rows = JSONArray()
    context.contentResolver
      .query(
        uri,
        proj,
        CalendarContract.Instances.CALENDAR_ID + "=?",
        arrayOf(calendarId().toString()),
        CalendarContract.Instances.BEGIN + " ASC",
      )
      ?.use { c ->
        while (c.moveToNext()) {
          val row =
            JSONObject()
              .put("title", c.getString(0) ?: "")
              .put("start", fmtMin.format(Date(c.getLong(1))))
              .put("end", fmtMin.format(Date(c.getLong(2))))
              .put("location", c.getString(3) ?: "")
          rows.put(row)
        }
      }
    return if (rows.length() == 0) "No events on ${date.trim().take(10)}" else rows.toString()
  }

  /**
   * The local day [date] ("2026-10-05") from its midnight to the next one, in ms: 23 or 25 hours on
   * a day the clocks change. Null for text that is not such a day.
   */
  internal fun dayWindow(date: String, zone: TimeZone = TimeZone.getDefault()): Pair<Long, Long>? {
    val format =
      SimpleDateFormat("yyyy-MM-dd", Locale.US).apply {
        isLenient = false
        timeZone = zone
      }
    val day = parseWhole(format, date.trim().take(10)) ?: return null
    val next =
      Calendar.getInstance(zone).apply {
        time = day
        add(Calendar.DAY_OF_YEAR, 1)
        set(Calendar.HOUR_OF_DAY, 0)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
      }
    return day.time to next.timeInMillis
  }

  /**
   * "2026-10-05 10:00", also with a T and with seconds (dropped), on this phone's clock. Any other
   * text, such as "73:00", "2026-02-30" or "9:30 PM", goes back to the model with the format.
   */
  internal fun parseMinute(s: String): Long {
    val t = s.trim().replace('T', ' ').replace(SECONDS, "$1")
    return parseWhole(fmtMin, t)?.time
      ?: throw IllegalArgumentException("bad time '$s' (use YYYY-MM-DD HH:MM)")
  }

  /**
   * [text] read by [format] to its end, or null: `DateFormat.parse` stops at the first character
   * it cannot read and returns what it read up to there (" PM" of "9:30 PM" is left over).
   */
  private fun parseWhole(format: SimpleDateFormat, text: String): Date? {
    val at = ParsePosition(0)
    val parsed = format.parse(text, at)
    return if (parsed != null && at.index == text.length) parsed else null
  }

  private fun addEvent(title: String, start: String, end: String, location: String): String {
    val s = parseMinute(start)
    val e = parseMinute(end)
    require(e > s) { "end must be after start" }
    val v =
      ContentValues().apply {
        put(CalendarContract.Events.CALENDAR_ID, calendarId())
        put(CalendarContract.Events.TITLE, title)
        put(CalendarContract.Events.EVENT_LOCATION, location)
        put(CalendarContract.Events.DTSTART, s)
        put(CalendarContract.Events.DTEND, e)
        put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().id)
      }
    context.contentResolver.insert(CalendarContract.Events.CONTENT_URI, v)
      ?: throw IllegalStateException("calendar refused the event")
    return receipt(title, s, e, location)
  }

  /** What add_calendar_event says it did: "… at <location>" only when there is a location. */
  internal fun receipt(title: String, start: Long, end: Long, location: String): String {
    val from = fmtMin.format(Date(start))
    val to = fmtMin.format(Date(end))
    val place = if (location.isBlank()) "" else " at $location"
    return "Event '$title' added: $from to $to$place"
  }

  /** The local "Phone Agent" calendar's id; the calendar is created on the first read or write. */
  private fun calendarId(): Long {
    val name = PhoneTools.CALENDAR_NAME
    val selection =
      CalendarContract.Calendars.ACCOUNT_TYPE + "=? AND " +
        CalendarContract.Calendars.ACCOUNT_NAME + "=? AND " +
        CalendarContract.Calendars.NAME + "=?"
    context.contentResolver
      .query(
        CalendarContract.Calendars.CONTENT_URI,
        arrayOf(CalendarContract.Calendars._ID),
        selection,
        arrayOf(CalendarContract.ACCOUNT_TYPE_LOCAL, name, name),
        null,
      )
      ?.use { c ->
        if (c.moveToFirst()) {
          return c.getLong(0)
        }
      }
    val uri =
      CalendarContract.Calendars.CONTENT_URI.buildUpon()
        .appendQueryParameter(CalendarContract.CALLER_IS_SYNCADAPTER, "true")
        .appendQueryParameter(CalendarContract.Calendars.ACCOUNT_NAME, name)
        .appendQueryParameter(
          CalendarContract.Calendars.ACCOUNT_TYPE,
          CalendarContract.ACCOUNT_TYPE_LOCAL,
        )
        .build()
    val v =
      ContentValues().apply {
        put(CalendarContract.Calendars.ACCOUNT_NAME, name)
        put(CalendarContract.Calendars.ACCOUNT_TYPE, CalendarContract.ACCOUNT_TYPE_LOCAL)
        put(CalendarContract.Calendars.NAME, name)
        put(CalendarContract.Calendars.CALENDAR_DISPLAY_NAME, name)
        put(CalendarContract.Calendars.CALENDAR_COLOR, 0xFF58A6FF.toInt())
        put(
          CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL,
          CalendarContract.Calendars.CAL_ACCESS_OWNER,
        )
        put(CalendarContract.Calendars.OWNER_ACCOUNT, name)
        put(CalendarContract.Calendars.VISIBLE, 1)
        put(CalendarContract.Calendars.SYNC_EVENTS, 1)
      }
    val row =
      context.contentResolver.insert(uri, v)
        ?: throw IllegalStateException("could not create a calendar")
    return ContentUris.parseId(row)
  }
}
