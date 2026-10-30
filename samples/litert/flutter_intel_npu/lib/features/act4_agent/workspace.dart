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

import 'dart:convert';

/// The world the assistant works in, apart from any one meeting: who is who,
/// which addresses are the company's own, the calendars, the shared
/// documents and whatever else an assistant would look up (open tickets, a
/// changelog). A real assistant reads these from the directory and the
/// calendar; here they come from a JSON file (`ACT4_WORKSPACE`).
///
/// It holds no answers: nothing in it says what a meeting decided.
///
/// The file is read strictly ([Workspace.fromJson]): an unknown key, a wrong
/// type or an impossible time is an error naming the field, because a typo
/// that silently drops a calendar or a domain also drops the rule built on it.
class Workspace {
  const Workspace({
    this.company,
    this.today,
    this.internalDomain,
    this.people = const [],
    this.calendars = const [],
    this.workHours = defaultWorkHours,
    this.documents = const [],
    this.notes = '',
    this.source,
  });

  /// No workspace: the agent knows only the meeting.
  static const empty = Workspace();

  /// Used when the file has no `work_hours`.
  static const defaultWorkHours = (start: 9 * 60, end: 18 * 60);

  /// The keys a workspace file may have.
  static const keys = [
    'company',
    'today',
    'internal_domain',
    'work_hours',
    'people',
    'calendars',
    'documents',
    'notes',
  ];

  final String? company;

  /// The day the assistant works on; null = the machine's date.
  final DateTime? today;

  /// Mail to any other domain must be a draft (the external-mail rule).
  /// Lowercase, without "@".
  final String? internalDomain;

  final List<Person> people;
  final List<CalendarDay> calendars;

  /// Minutes from midnight: meetings are booked inside these.
  final ({int start, int end}) workHours;

  /// Documents the assistant may write to with `update_doc`.
  final List<String> documents;

  /// Markdown passed to the model as is: tickets, changelogs, context.
  final String notes;

  /// The file it was read from; null for none and for the bundled example.
  final String? source;

  bool get isEmpty => people.isEmpty && calendars.isEmpty && documents.isEmpty && notes.isEmpty;

  Workspace copyWith({String? source}) => Workspace(
    company: company,
    today: today,
    internalDomain: internalDomain,
    people: people,
    calendars: calendars,
    workHours: workHours,
    documents: documents,
    notes: notes,
    source: source ?? this.source,
  );

  /// Null when nobody's calendar for [date] is known.
  CalendarDay? calendarOn(String date) {
    for (final c in calendars) {
      if (c.date == date) return c;
    }
    return null;
  }

  bool isInternal(String address) {
    final d = internalDomain;
    if (d == null) return true;
    return address.trim().toLowerCase().endsWith('@${d.toLowerCase()}');
  }

  /// [json] is the file's text. Problems are thrown as [FormatException]
  /// naming the field (`workspace.calendars[0].busy.ana@x.io`).
  factory Workspace.fromJson(String json, {String? source}) {
    final Object? root;
    try {
      // Notepad's UTF-8 starts with a byte order mark; JSON may not.
      root = jsonDecode(json.startsWith('\uFEFF') ? json.substring(1) : json);
    } on FormatException catch (e) {
      throw FormatException('workspace: not JSON (${e.message})');
    }
    final j = _object(root, 'workspace', keys);
    String? str(String k) => _string(j[k], 'workspace.$k', optional: true);
    final domain = str('internal_domain')?.trim().toLowerCase();
    if (domain != null && (domain.isEmpty || domain.contains('@') || domain.contains(RegExp(r'\s')))) {
      throw FormatException('workspace.internal_domain: expected a domain like "example.com", got "$domain"');
    }
    final today = str('today');
    final hours = j['work_hours'];
    final workHours = hours == null ? defaultWorkHours : _span(hours, 'workspace.work_hours');
    final people = [
      for (final (i, p) in _list(j['people'], 'workspace.people').indexed) Person._fromJson(p, 'people[$i]', domain),
    ];
    _unique(people.map((p) => p.email), 'workspace.people', 'email');
    final calendars = [
      for (final (i, c) in _list(j['calendars'], 'workspace.calendars').indexed)
        CalendarDay._fromJson(c, 'calendars[$i]'),
    ];
    _unique(calendars.map((c) => c.date), 'workspace.calendars', 'date');
    return Workspace(
      company: str('company'),
      today: today == null ? null : _date(today, 'workspace.today'),
      internalDomain: domain,
      people: people,
      calendars: calendars,
      workHours: workHours,
      documents: [
        for (final (i, d) in _list(j['documents'], 'workspace.documents').indexed)
          _string(d, 'workspace.documents[$i]', nonEmpty: true)!,
      ],
      notes: str('notes') ?? '',
      source: source,
    );
  }

  /// The workspace as the model reads it, in the system message.
  String describe() {
    final b = StringBuffer();
    if (people.isNotEmpty) {
      b.writeln('## People\n');
      for (final internal in [true, false]) {
        final group = people.where((p) => p.internal == internal).toList();
        if (group.isEmpty) continue;
        b.writeln(internal ? 'Internal:' : 'External:');
        for (final p in group) {
          b.writeln('- ${p.name}${p.role.isEmpty ? '' : ', ${p.role}'} — ${p.email}');
        }
        b.writeln();
      }
    }
    for (final c in calendars) {
      b.writeln('## Calendar, ${weekday(c.date)} ${c.date} (busy times; everyone is free otherwise)\n');
      for (final MapEntry(key: who, value: spans) in c.busy.entries) {
        final name = people.where((p) => p.email == who).map((p) => p.name).firstOrNull ?? who;
        b.writeln(
          '- $name: ${spans == null ? 'out of office' : spans.map((s) => '${hm(s.$1)}–${hm(s.$2)}').join(', ')}',
        );
      }
      b.writeln();
    }
    if (documents.isNotEmpty) {
      b.writeln('## Shared documents\n');
      for (final d in documents) {
        b.writeln('- "$d"');
      }
      b.writeln();
    }
    if (notes.trim().isNotEmpty) b.writeln(notes.trim());
    return b.toString().trim();
  }
}

class Person {
  const Person({required this.name, required this.email, this.role = '', this.internal = true});

  static const keys = ['name', 'email', 'role', 'internal'];

  final String name;

  /// Lowercase.
  final String email;
  final String role;
  final bool internal;

  /// [domain] is the workspace's internal domain: a person's `internal` must
  /// agree with it, or the people list and the mail rule would disagree.
  factory Person._fromJson(Object? json, String at, String? domain) {
    final j = _object(json, 'workspace.$at', keys);
    final email = _address(j['email'], 'workspace.$at.email');
    final internal = j['internal'] ?? true;
    if (internal is! bool) throw FormatException('workspace.$at.internal: expected true or false, got $internal');
    if (domain != null && internal != email.endsWith('@$domain')) {
      throw FormatException(
        internal
            ? 'workspace.$at: $email is outside internal_domain "$domain" but internal; add "internal": false'
            : 'workspace.$at: $email is in internal_domain "$domain" but marked "internal": false',
      );
    }
    return Person(
      name: _string(j['name'], 'workspace.$at.name', nonEmpty: true)!,
      email: email,
      role: _string(j['role'], 'workspace.$at.role', optional: true) ?? '',
      internal: internal,
    );
  }
}

/// Busy times on one day. A person with null spans is out of office; a
/// person not listed is free all day.
class CalendarDay {
  const CalendarDay({required this.date, required this.busy});

  static const keys = ['date', 'busy'];

  /// YYYY-MM-DD.
  final String date;

  /// Address (lowercase) → busy spans in minutes from midnight, or null (out
  /// of office).
  final Map<String, List<(int, int)>?> busy;

  factory CalendarDay._fromJson(Object? json, String at) {
    final j = _object(json, 'workspace.$at', keys);
    final date = _string(j['date'], 'workspace.$at.date')!;
    _date(date, 'workspace.$at.date');
    final busy = j['busy'];
    if (busy is! Map) {
      throw FormatException('workspace.$at.busy: expected {address: [["HH:MM", "HH:MM"], …] | "out"}');
    }
    final out = <String, List<(int, int)>?>{};
    for (final MapEntry(:key, :value) in busy.entries) {
      final who = _address(key, 'workspace.$at.busy key');
      final path = 'workspace.$at.busy.$who';
      if (out.containsKey(who)) throw FormatException('$path: listed twice');
      out[who] = switch (value) {
        'out' => null,
        List() => [
          for (final (i, s) in value.indexed)
            if (_span(s, '$path[$i]') case (:final start, :final end)) (start, end),
        ],
        _ => throw FormatException('$path: expected [["HH:MM", "HH:MM"], …] or "out", got ${jsonEncode(value)}'),
      };
    }
    return CalendarDay(date: date, busy: out);
  }

  /// Why [who] cannot meet at [start]..[end], or null if free.
  String? clash(String who, int start, int end) {
    if (!busy.containsKey(who)) return null;
    final spans = busy[who];
    if (spans == null) return 'out of office';
    for (final (b, e) in spans) {
      if (start < e && b < end) return 'busy ${hm(b)}–${hm(e)}';
    }
    return null;
  }
}

String hm(int minutes) => '${(minutes ~/ 60).toString().padLeft(2, '0')}:${(minutes % 60).toString().padLeft(2, '0')}';

String ymd(DateTime d) =>
    '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';

String weekday(String date) {
  final d = DateTime.tryParse(date);
  if (d == null) return '';
  return const ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][d.weekday - 1];
}

// The readers below check types with `is` and throw a FormatException naming
// the field: a cast would throw a TypeError that names neither.

Map<Object?, Object?> _object(Object? v, String path, List<String> keys) {
  if (v is! Map) throw FormatException('$path: expected an object with ${keys.join(', ')}');
  for (final k in v.keys) {
    if (!keys.contains(k)) throw FormatException('$path.$k: unknown key, expected ${keys.join(', ')}');
  }
  return v;
}

List<Object?> _list(Object? v, String path) {
  if (v == null) return const [];
  if (v is! List) throw FormatException('$path: expected a list');
  return v;
}

String? _string(Object? v, String path, {bool optional = false, bool nonEmpty = false}) {
  if (v == null && optional) return null;
  if (v is! String) throw FormatException('$path: expected a string${v == null ? '' : ', got ${jsonEncode(v)}'}');
  if (nonEmpty && v.trim().isEmpty) throw FormatException('$path: empty');
  return v;
}

final _addressPattern = RegExp(r'^[^@\s]+@[^@\s]+$');

String _address(Object? v, String path) {
  final s = _string(v, path)!.trim().toLowerCase();
  if (!_addressPattern.hasMatch(s)) throw FormatException('$path: expected an email address, got "$s"');
  return s;
}

/// YYYY-MM-DD, a day that exists.
DateTime _date(String s, String path) {
  final m = RegExp(r'^(\d{4})-(\d\d)-(\d\d)$').firstMatch(s);
  final d = m == null ? null : DateTime(int.parse(m[1]!), int.parse(m[2]!), int.parse(m[3]!));
  if (d == null || ymd(d) != s) throw FormatException('$path: expected a date YYYY-MM-DD, got "$s"');
  return d;
}

/// "HH:MM", 00:00 to 23:59, as minutes from midnight.
int _minutes(Object? v, String path) {
  final m = v is String ? RegExp(r'^(\d\d):(\d\d)$').firstMatch(v) : null;
  final h = m == null ? 99 : int.parse(m[1]!), min = m == null ? 99 : int.parse(m[2]!);
  if (h > 23 || min > 59) throw FormatException('$path: expected a time "HH:MM" (00:00–23:59), got ${jsonEncode(v)}');
  return h * 60 + min;
}

/// ["HH:MM", "HH:MM"], the start before the end.
({int start, int end}) _span(Object? v, String path) {
  if (v is! List || v.length != 2) {
    throw FormatException('$path: expected ["HH:MM", "HH:MM"], got ${jsonEncode(v)}');
  }
  final start = _minutes(v[0], '$path[0]'), end = _minutes(v[1], '$path[1]');
  if (start >= end) throw FormatException('$path: ${v[0]} is not before ${v[1]}');
  return (start: start, end: end);
}

void _unique(Iterable<String> values, String path, String what) {
  final seen = <String>{};
  for (final v in values) {
    if (!seen.add(v)) throw FormatException('$path: $what $v appears twice');
  }
}
