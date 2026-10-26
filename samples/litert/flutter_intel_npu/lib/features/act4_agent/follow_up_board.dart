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
import 'dart:math' as math;

import 'agent_run.dart';
import 'meeting_task.dart';
import 'workspace.dart';

/// One follow-up the model found in the meeting.
class FollowUp {
  const FollowUp({required this.id, required this.what, required this.who, required this.due});

  /// The model's own number for it; a missing one is numbered after the
  /// largest given. Two follow-ups can share one (the model's mistake, which
  /// the board reports).
  final int id;
  final String what;
  final String who;

  /// YYYY-MM-DD, or empty.
  final String due;

  @override
  bool operator ==(Object other) =>
      other is FollowUp && other.id == id && other.what == what && other.who == who && other.due == due;

  @override
  int get hashCode => Object.hash(id, what, who, due);
}

/// One line of the checklist: a follow-up (done once an action names it) or
/// a rule (kept unless an action breaks it).
class BoardItem {
  const BoardItem({
    required this.id,
    required this.label,
    required this.passed,
    this.detail = '',
    this.why = '',
    this.rule = false,
  });

  /// Unique on the board: `f<number>` for a follow-up (`f<number>-2`, … for
  /// a number the model gave twice), the rule's name for a rule.
  final String id;
  final String label;

  /// Who and by when, for a follow-up. For a rule: what it could not check
  /// (e.g. a meeting on a day with no calendar), shown whether it passed or
  /// not.
  final String detail;
  final bool passed;

  /// Why it failed, in words.
  final String why;
  final bool rule;
}

/// The checklist of a run, built from the run alone: the list the model
/// published and the actions it took. No item comes from the app.
///
/// The score is "done of what it listed", not "right against an answer key":
/// the model writes its own list, so a list that misses a follow-up costs
/// nothing here. The first list it publishes is the one scored; publishing a
/// different one later is a broken rule, not a new checklist. The rules are
/// checked against the workspace the run worked in.
abstract final class FollowUpBoard {
  /// What the board reads from the calls: tool → argument paths (`a.b` is
  /// field `b` of the items of array `a`). A test pins them to the schemas.
  static const reads = {
    MeetingTask.listTool: {_items, '$_items.$_id', '$_items.$_what', '$_items.$_who', '$_items.$_due'},
    MeetingTask.sendEmail: {_to, MeetingTask.itemArg},
    MeetingTask.askClarification: {_to, MeetingTask.itemArg},
    MeetingTask.scheduleMeeting: {_attendees, _date, _start, _duration, MeetingTask.itemArg},
    MeetingTask.createTask: {MeetingTask.itemArg},
    MeetingTask.draftEmail: {MeetingTask.itemArg},
    MeetingTask.updateDoc: {MeetingTask.itemArg},
  };

  static const _items = 'items';
  static const _id = 'id';
  static const _what = 'what';
  static const _who = 'who';
  static const _due = 'due';
  static const _to = 'to';
  static const _attendees = 'attendees';
  static const _date = 'date';
  static const _start = 'start';
  static const _duration = 'duration_min';

  /// The first list the model published, or empty before it does.
  static List<FollowUp> listOf(AgentRun? run) => _lists(run).firstOrNull?.$2 ?? const [];

  /// Every list call whose items read as a list, in order, with its call.
  static List<(AgentCall, List<FollowUp>)> _lists(AgentRun? run) => [
    for (final c in run?.calls ?? const <AgentCall>[])
      if (c.tool == MeetingTask.listTool)
        if (_itemsOf(c.args[_items]) case final items?) (c, _followUps(items)),
  ];

  /// `items` as a list; one written as a JSON string is read too.
  static List<Object?>? _itemsOf(Object? v) {
    if (v is List) return v;
    if (v is! String) return null;
    try {
      final d = jsonDecode(v);
      return d is List ? d : null;
    } on FormatException {
      return null;
    }
  }

  static List<FollowUp> _followUps(List<Object?> items) {
    final maps = [
      for (final it in items)
        if (it is Map) it,
    ];
    var next = [for (final it in maps) ?_int(it[_id])].fold(0, math.max);
    return [
      for (final it in maps)
        FollowUp(
          id: _int(it[_id]) ?? ++next,
          what: '${it[_what] ?? ''}'.trim(),
          who: '${it[_who] ?? ''}'.trim(),
          due: '${it[_due] ?? ''}'.trim(),
        ),
    ];
  }

  /// The actions: every executed call but the list.
  static List<AgentCall> actionsOf(AgentRun? run) => [
    for (final c in run?.calls ?? const <AgentCall>[])
      if (c.tool != MeetingTask.listTool) c,
  ];

  /// The follow-up an action names, or null.
  static int? itemOf(AgentCall c) => _int(c.args[MeetingTask.itemArg]);

  /// The checklist of [run]: the follow-ups it listed, then the rules,
  /// checked against [w], the workspace the run works in.
  static List<BoardItem> of(AgentRun? run, Workspace w) {
    final list = listOf(run);
    final actions = actionsOf(run);
    return [..._followUpItems(list, actions), ...rules(list, actions, w, lists: _lists(run))];
  }

  static List<BoardItem> _followUpItems(List<FollowUp> list, List<AgentCall> actions) {
    final done = {for (final a in actions) ?itemOf(a)};
    final seen = <int, int>{};
    final out = <BoardItem>[];
    for (final f in list) {
      final n = seen[f.id] = (seen[f.id] ?? 0) + 1;
      out.add(
        BoardItem(
          id: n == 1 ? 'f${f.id}' : 'f${f.id}-$n',
          label: f.what.isEmpty ? 'Follow-up ${f.id}' : f.what,
          detail: [if (f.who.isNotEmpty) f.who, if (f.due.isNotEmpty) f.due].join(' · '),
          // An action ticks the first follow-up with its number only.
          passed: n == 1 && done.contains(f.id),
          why: n == 1 ? 'no action for it' : 'its number ${f.id} was given twice; an action ticks the first',
        ),
      );
    }
    return out;
  }

  /// The rules that the workspace lets the app check. One the workspace says
  /// nothing about (no internal domain, no calendar) is left out, not passed.
  /// The two about the list itself (its numbers, publishing it once) show
  /// only when broken: they are the model's slip, not a rule it was given.
  static List<BoardItem> rules(
    List<FollowUp> list,
    List<AgentCall> actions,
    Workspace w, {
    List<(AgentCall, List<FollowUp>)> lists = const [],
  }) {
    final out = <BoardItem>[];
    if (w.internalDomain != null) out.add(_externalRule(actions, w));
    if (w.calendars.isNotEmpty) out.add(_calendarRule(actions, w));
    final ids = {for (final f in list) f.id};
    final stray = actions.where((a) => !ids.contains(itemOf(a))).length;
    out.add(
      BoardItem(
        id: 'actions_on_the_list',
        label: 'Every action is for a listed follow-up',
        passed: stray == 0,
        why: '$stray action${stray == 1 ? '' : 's'} not on its own list',
        rule: true,
      ),
    );
    final seen = <String>{};
    var repeated = 0;
    for (final a in actions) {
      final args = Map.of(a.args)..remove(MeetingTask.itemArg);
      if (!seen.add('${a.tool} ${jsonEncode(_canonical(args))}')) repeated++;
    }
    out.add(
      BoardItem(
        id: 'no_repeated_actions',
        label: 'No action done twice',
        passed: repeated == 0,
        why: '$repeated repeated',
        rule: true,
      ),
    );
    final counts = <int, int>{};
    for (final f in list) {
      counts[f.id] = (counts[f.id] ?? 0) + 1;
    }
    final shared = [
      for (final MapEntry(key: id, value: n) in counts.entries)
        if (n > 1) 'number $id is given to $n follow-ups',
    ];
    if (shared.isNotEmpty) {
      out.add(
        BoardItem(
          id: 'unique_numbers',
          label: 'Every follow-up has its own number',
          passed: false,
          why: shared.join('; '),
          rule: true,
        ),
      );
    }
    if (lists.isNotEmpty) {
      final first = lists.first.$2;
      final later = [
        for (final (_, l) in lists.skip(1))
          if (!_sameList(first, l)) l,
      ];
      if (later.isNotEmpty) {
        out.add(
          BoardItem(
            id: 'listed_once',
            label: 'The list is published once',
            passed: false,
            why:
                '${later.length == 1 ? 'a later list differs' : '${later.length} later lists differ'}: '
                '${_change(first, later.first)}; the first one is scored',
            rule: true,
          ),
        );
      }
    }
    return out;
  }

  /// Mail and questions go to outside addresses only as drafts.
  static BoardItem _externalRule(List<AgentCall> actions, Workspace w) {
    final broken = <String>[];
    for (final a in actions) {
      final verb = switch (a.tool) {
        MeetingTask.sendEmail => 'emailed',
        MeetingTask.askClarification => 'asked',
        _ => null,
      };
      if (verb == null) continue;
      final (:addresses, :bad) = recipients(a.args[_to]);
      final outside = [
        for (final to in addresses)
          if (!w.isInternal(to)) to,
      ];
      if (outside.isNotEmpty) broken.add('$verb ${outside.join(', ')} directly instead of a draft');
      for (final b in bad) {
        broken.add('$verb "$b", which is not an email address');
      }
    }
    return BoardItem(
      id: 'external_as_drafts',
      label: 'Nothing sent straight to an outside address',
      passed: broken.isEmpty,
      why: broken.toSet().join('; '),
      rule: true,
    );
  }

  /// Meetings inside working hours, where every attendee is free. What the
  /// workspace cannot say (a day without a calendar, an attendee not in the
  /// directory or not an address) is not a clash: it is named in the rule's
  /// detail as not checked.
  static BoardItem _calendarRule(List<AgentCall> actions, Workspace w) {
    final clashes = <String>[];
    final unchecked = <String>[];
    final known = {for (final p in w.people) p.email.trim().toLowerCase()};
    for (final a in actions.where((a) => a.tool == MeetingTask.scheduleMeeting)) {
      final date = '${a.args[_date] ?? ''}'.trim();
      final startArg = a.args[_start];
      final start = startArg is String ? _minutes(startArg) : null;
      final dur = _int(a.args[_duration]);
      if (start == null) {
        clashes.add(startArg == null ? 'a meeting has no start time' : '"$startArg" is not a time (HH:MM)');
        continue;
      }
      if (dur == null || dur <= 0) {
        clashes.add('the meeting at ${hm(start)} has no length');
        continue;
      }
      final end = start + dur;
      if (start < w.workHours.start) clashes.add('${hm(start)} is before working hours');
      if (end > w.workHours.end) clashes.add('${hm(start)}–${hm(end)} ends after working hours');
      if (!RegExp(r'^\d{4}-\d\d-\d\d$').hasMatch(date) || DateTime.tryParse(date) == null) {
        clashes.add('"$date" is not a date (YYYY-MM-DD)');
        continue;
      }
      final (:addresses, :bad) = recipients(a.args[_attendees]);
      for (final b in bad) {
        unchecked.add('"$b" (not an address)');
      }
      final day = w.calendarOn(date);
      if (day == null) {
        unchecked.add('$date (no calendar)');
        continue;
      }
      for (final who in addresses) {
        if (known.isNotEmpty && !known.contains(who)) {
          unchecked.add('$who (not in the directory)');
          continue;
        }
        final c = day.clash(who, start, end);
        if (c != null) clashes.add('${who.split('@').first} is $c');
      }
    }
    return BoardItem(
      id: 'meetings_in_free_slots',
      label: 'Meetings booked where everyone is free',
      passed: clashes.isEmpty,
      why: clashes.join('; '),
      detail: unchecked.isEmpty ? '' : 'not checked: ${unchecked.toSet().join(', ')}',
      rule: true,
    );
  }

  /// The addresses in a recipient argument, lowercased, and the entries that
  /// are not addresses. It may be a list or one string; entries are split at
  /// commas and semicolons, and `Name <x@y>` reads as `x@y`.
  static ({List<String> addresses, List<String> bad}) recipients(Object? v) {
    final entries = [
      for (final e in v is List ? v : (v == null ? const [] : [v]))
        for (final part in '$e'.split(RegExp('[,;]')))
          if (part.trim().isNotEmpty) part.trim(),
    ];
    final addresses = <String>[];
    final bad = <String>[];
    for (final e in entries) {
      final address = RegExp(r'<([^<>]*)>\s*$').firstMatch(e)?.group(1)?.trim() ?? e;
      if (RegExp(r'^[^\s@<>]+@[^\s@<>]+$').hasMatch(address)) {
        addresses.add(address.toLowerCase());
      } else {
        bad.add(e);
      }
    }
    return (addresses: addresses, bad: bad);
  }

  static bool _sameList(List<FollowUp> a, List<FollowUp> b) =>
      a.length == b.length && [for (var i = 0; i < a.length; i++) a[i] == b[i]].every((same) => same);

  /// How [later] differs from [first], by number.
  static String _change(List<FollowUp> first, List<FollowUp> later) {
    final a = {for (final f in first.reversed) f.id: f};
    final b = {for (final f in later.reversed) f.id: f};
    final added = [
      for (final id in b.keys)
        if (!a.containsKey(id)) id,
    ]..sort();
    final dropped = [
      for (final id in a.keys)
        if (!b.containsKey(id)) id,
    ]..sort();
    final changed = [
      for (final id in a.keys)
        if (b.containsKey(id) && a[id] != b[id]) id,
    ]..sort();
    final parts = [
      if (added.isNotEmpty) 'added ${added.join(', ')}',
      if (dropped.isNotEmpty) 'dropped ${dropped.join(', ')}',
      if (changed.isNotEmpty) 'changed ${changed.join(', ')}',
    ];
    return parts.isEmpty ? 'the same follow-ups in another order or count' : parts.join('; ');
  }

  /// [v] with every object's keys sorted, so two calls that differ only in
  /// key order compare equal.
  static Object? _canonical(Object? v) => switch (v) {
    Map() => {
      for (final k in [for (final k in v.keys) '$k']..sort()) k: _canonical(v[k]),
    },
    List() => [for (final e in v) _canonical(e)],
    _ => v,
  };

  static int? _int(Object? v) => switch (v) {
    int() => v,
    double() when v == v.roundToDouble() => v.toInt(),
    String() => int.tryParse(v.trim().replaceFirst('#', '')),
    _ => null,
  };

  /// Minutes from midnight of a valid 24-hour `HH:MM`, else null.
  static int? _minutes(String s) {
    final m = RegExp(r'^(\d{1,2}):(\d\d)$').firstMatch(s.trim());
    if (m == null) return null;
    final (h, min) = (int.parse(m.group(1)!), int.parse(m.group(2)!));
    return h < 24 && min < 60 ? h * 60 + min : null;
  }
}
