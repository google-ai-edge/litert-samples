// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import 'meeting_input.dart';
import 'workspace.dart';

/// The follow-up of one meeting, in one request with two steps: the model
/// first lists the follow-ups it finds in the transcript (that list is the
/// checklist), then carries them out, naming the item each action is for.
///
/// One request, not two: on the NPU a second message after a tool loop
/// fails (the executor rolls back one token at most), and the model takes the
/// two steps as separate turns anyway.
///
/// Nothing here knows the meeting: the transcript is the input, and the
/// workspace says who is who, which mail is internal and what the calendars
/// hold.
class MeetingTask {
  MeetingTask({required this.meeting, required this.workspace}) : system = _system(workspace), user = _user(meeting);

  final MeetingInput meeting;
  final Workspace workspace;

  final String system;

  /// The request: list the follow-ups, then carry them out.
  final String user;

  /// The tool that publishes the list; the screen builds the checklist from it.
  static const listTool = 'list_follow_ups';

  /// The argument every action carries: the number of its item.
  static const itemArg = 'follow_up';

  // The six actions. The checklist reads some of them by name, so the names
  // live here once.
  static const createTask = 'create_task';
  static const sendEmail = 'send_email';
  static const draftEmail = 'draft_email';
  static const scheduleMeeting = 'schedule_meeting';
  static const updateDoc = 'update_doc';
  static const askClarification = 'ask_clarification';

  static String _system(Workspace w) {
    final today = w.today ?? DateTime.now();
    final date = today.toIso8601String().substring(0, 10);
    // Without a workspace the transcript is all there is: its own date says
    // when "Friday" is, and its names are the only way to name people.
    final when = w.today != null
        ? 'Today is ${weekday(date)} $date.'
        : 'Work out dates from the date of the meeting in the transcript; if it gives none, today is '
              '${weekday(date)} $date.';
    final domain = w.internalDomain;
    final rules = [
      if (domain != null)
        'You may send email only to @$domain addresses. Anything for an external address must be a $draftEmail, '
            'approved by the person who owns that relationship.'
      else
        'Anything for someone outside the company must be a $draftEmail, approved by the person who owns that '
            'relationship.',
      'You cannot make phone calls, sign documents, place orders or approve spending. Create a task for the person '
          'who will do it.',
      'Do not create tasks for work the meeting decided against, for work that is already done or already tracked, '
          'or for topics declared out of scope.',
      'If the meeting left an owner undecided, ask the people who said they would decide it, instead of guessing.',
      if (w.calendars.isNotEmpty)
        'Book meetings only in slots where every attendee is free (see the calendar), between '
            '${hm(w.workHours.start)} and ${hm(w.workHours.end)}.'
      else
        'Book meetings between ${hm(w.workHours.start)} and ${hm(w.workHours.end)}.',
      'Use absolute dates (YYYY-MM-DD). Business days are Monday to Friday.',
      if (w.people.isNotEmpty)
        'Address people by the email addresses listed under People.'
      else
        'You have no directory: name people as the transcript names them, and use an email address only if the '
            'transcript gives it.',
    ];
    final who = w.company == null ? 'the team assistant' : 'the team assistant of ${w.company}';
    final context = w.describe();
    return 'You are $who. A meeting has just ended. Read the transcript you are given, work out every follow-up '
        'action, and carry out the follow-up: do what you can do yourself with your tools, and hand everything '
        'else to the right person. $when\n\n'
        '## Rules\n\n${[for (final (i, r) in rules.indexed) '${i + 1}. $r'].join('\n')}'
        '${context.isEmpty ? '' : '\n\n$context'}';
  }

  static String _user(MeetingInput m) =>
      'Here is the transcript of the meeting that just ended.\n\n'
      'Step 1: call $listTool once with every follow-up the meeting needs: each piece of work someone must do, each '
      'message to send, each meeting to book, each document to update, each open question to ask. Number them '
      'from 1. Leave out what the meeting decided against, what is already done or tracked, and what is out of '
      'scope.\n\n'
      'Step 2: then carry the follow-ups out with your tools: do what you can do yourself and hand everything else '
      'to the right person. Give every action, in `$itemArg`, the number that follow-up has in your own list. '
      'When you are done, reply with a short summary.\n\n${m.text}';

  static const _s = {'type': 'string'};
  static const _item = {'type': 'integer', 'description': 'The number of the follow-up this action is for.'};

  /// The listing tool and the six actions. The actions only record the call:
  /// nothing is sent or booked.
  List<Tool> get tools => [
    _tool(
      listTool,
      'Records the list of follow-ups the meeting needs, before any of them is carried out.',
      {
        'items': {
          'type': 'array',
          'items': {
            'type': 'object',
            'properties': {
              'id': {'type': 'integer', 'description': 'Number of the follow-up, from 1.'},
              'what': {..._s, 'description': 'What has to happen, in a few words.'},
              'who': {..._s, 'description': 'Who does it or receives it.'},
              'due': {..._s, 'description': 'YYYY-MM-DD, or empty if the meeting set no date.'},
            },
            'required': ['id', 'what', 'who'],
          },
        },
      },
      ['items'],
    ),
    _tool(
      createTask,
      'Assigns a piece of work to a person.',
      {
        'title': _s,
        'owner': {..._s, 'description': 'Email address of the person who will do it.'},
        'due': {..._s, 'description': 'Due date, YYYY-MM-DD.'},
        'depends_on': {'type': 'array', 'items': _s, 'description': 'Titles of tasks this one waits for.'},
        'notes': _s,
        itemArg: _item,
      },
      ['title', 'owner', 'due', itemArg],
    ),
    _tool(
      sendEmail,
      'Sends an email immediately.',
      {
        'to': {'type': 'array', 'items': _s},
        'subject': _s,
        'body': _s,
        itemArg: _item,
      },
      ['to', 'subject', 'body', itemArg],
    ),
    _tool(
      draftEmail,
      'Saves an email as a draft; the approver reviews it and sends it.',
      {
        'to': {'type': 'array', 'items': _s},
        'subject': _s,
        'body': _s,
        'approver': {..._s, 'description': 'Email address of the person who approves and sends it.'},
        itemArg: _item,
      },
      ['to', 'subject', 'body', 'approver', itemArg],
    ),
    _tool(
      scheduleMeeting,
      'Books a meeting.',
      {
        'title': _s,
        'attendees': {'type': 'array', 'items': _s},
        'date': {..._s, 'description': 'YYYY-MM-DD'},
        'start': {..._s, 'description': 'HH:MM, 24h'},
        'duration_min': {'type': 'integer'},
        itemArg: _item,
      },
      ['title', 'attendees', 'date', 'start', 'duration_min', itemArg],
    ),
    _tool(
      updateDoc,
      workspace.documents.isEmpty
          ? 'Writes content into a shared document.'
          : 'Writes content into a shared document: ${workspace.documents.map((d) => '"$d"').join(' or ')}.',
      {'doc': _s, 'content': _s, itemArg: _item},
      ['doc', 'content', itemArg],
    ),
    _tool(
      askClarification,
      'Asks a person a question; use it when you cannot act without their answer.',
      {
        'to': {..._s, 'description': 'Email address.'},
        'question': _s,
        itemArg: _item,
      },
      ['to', 'question', itemArg],
    ),
  ];

  static Tool _tool(String name, String description, Map<String, Object> properties, List<String> required) => Tool(
    name: name,
    description: description,
    parameters: {'type': 'object', 'properties': properties, 'required': required},
  );
}
