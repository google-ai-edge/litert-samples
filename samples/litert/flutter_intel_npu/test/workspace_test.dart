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
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

/// [patch] merged into a valid minimal workspace, so each test changes one thing.
Workspace _ws(Map<String, Object?> patch) => Workspace.fromJson(
  jsonEncode({
    'internal_domain': 'x.io',
    'people': [
      {'name': 'Ana', 'email': 'ana@x.io'},
    ],
    ...patch,
  }),
);

Matcher _error(String part) => throwsA(isA<FormatException>().having((e) => e.message, 'message', contains(part)));

Map<String, Object?> _day(Object? busy, {String date = '2026-11-18'}) => {
  'calendars': [
    {'date': date, 'busy': busy},
  ],
};

void main() {
  final example = File('assets/agent/example/nimbus.workspace.json').readAsStringSync();

  group('a valid workspace', () {
    test('the bundled example parses under the strict rules', () {
      final w = Workspace.fromJson(example);
      expect(w.people, hasLength(11));
      expect(w.workHours, (start: 9 * 60, end: 18 * 60));
      expect(w.today, DateTime(2026, 11, 3));
      expect(w.source, isNull);
    });

    test('source is carried by fromJson and copyWith', () {
      final w = Workspace.fromJson(example, source: '/a/ws.json');
      expect(w.source, '/a/ws.json');
      expect(w.copyWith(source: '/b.json').source, '/b.json');
      expect(w.copyWith(source: '/b.json').people, w.people);
      expect(Workspace.empty.source, isNull);
    });

    test('addresses, busy keys and the domain are lowercased, so mixed case still matches', () {
      final w = Workspace.fromJson(
        jsonEncode({
          'internal_domain': 'Nimbus.IO',
          'people': [
            {'name': 'Ana', 'email': 'Ana.Ruiz@Nimbus.io'},
          ],
          'calendars': [
            {
              'date': '2026-11-18',
              'busy': {
                'Ana.Ruiz@NIMBUS.io': [
                  ['09:00', '10:00'],
                ],
              },
            },
          ],
        }),
      );
      expect(w.internalDomain, 'nimbus.io');
      expect(w.people.single.email, 'ana.ruiz@nimbus.io');
      expect(w.calendarOn('2026-11-18')!.clash('ana.ruiz@nimbus.io', 9 * 60, 9 * 60 + 30), 'busy 09:00–10:00');
      expect(w.describe(), contains('- Ana: 09:00–10:00'));
    });

    test('a UTF-8 byte order mark before the JSON is accepted', () {
      expect(Workspace.fromJson('﻿{"company": "C"}').company, 'C');
    });
  });

  group('unknown keys are refused, naming where', () {
    test('top level', () {
      expect(() => _ws({'calender': []}), _error('workspace.calender: unknown key, expected company'));
    });
    test('person', () {
      expect(
        () => _ws({
          'people': [
            {'name': 'A', 'email': 'a@x.io', 'mail': 'a@x.io'},
          ],
        }),
        _error('workspace.people[0].mail: unknown key'),
      );
    });
    test('calendar day', () {
      expect(
        () => _ws({
          'calendars': [
            {'date': '2026-11-18', 'busy': {}, 'note': 'x'},
          ],
        }),
        _error('workspace.calendars[0].note: unknown key'),
      );
    });
  });

  group('wrong types are FormatExceptions naming the field, never TypeErrors', () {
    final cases = <String, Map<String, Object?>>{
      'workspace.company': {'company': 1},
      'workspace.internal_domain': {'internal_domain': true},
      'workspace.today': {'today': 20261103},
      'workspace.notes': {
        'notes': ['a'],
      },
      'workspace.people': {
        'people': {'name': 'A'},
      },
      'workspace.people[0]': {
        'people': ['ana@x.io'],
      },
      'workspace.people[0].name': {
        'people': [
          {'name': 3, 'email': 'a@x.io'},
        ],
      },
      'workspace.people[0].role': {
        'people': [
          {'name': 'A', 'email': 'a@x.io', 'role': 1},
        ],
      },
      'workspace.people[0].internal': {
        'people': [
          {'name': 'A', 'email': 'a@x.io', 'internal': 'no'},
        ],
      },
      'workspace.calendars': {'calendars': 'monday'},
      'workspace.calendars[0].date': {
        'calendars': [
          {'busy': {}},
        ],
      },
      'workspace.calendars[0].busy': _day(['a@x.io']),
      'workspace.documents[0]': {
        'documents': [1],
      },
      'workspace.work_hours': {'work_hours': '09:00-18:00'},
    };
    for (final MapEntry(key: path, value: patch) in cases.entries) {
      test(path, () => expect(() => _ws(patch), _error(path)));
    }
    test('not JSON, not an object', () {
      expect(() => Workspace.fromJson('{'), _error('workspace: not JSON'));
      expect(() => Workspace.fromJson('[1]'), _error('workspace: expected an object'));
    });
  });

  group('times', () {
    test('work_hours: exactly two "HH:MM", start before end', () {
      expect(
        () => _ws({
          'work_hours': ['09:00'],
        }),
        _error('workspace.work_hours: expected ["HH:MM", "HH:MM"]'),
      );
      expect(
        () => _ws({
          'work_hours': ['09:00', '12:00', '18:00'],
        }),
        _error('workspace.work_hours'),
      );
      expect(
        () => _ws({
          'work_hours': ['18:00', '09:00'],
        }),
        _error('workspace.work_hours: 18:00 is not before 09:00'),
      );
      expect(
        () => _ws({
          'work_hours': ['09:00', '09:00'],
        }),
        _error('is not before'),
      );
      expect(
        () => _ws({
          'work_hours': ['9:00', '18:00'],
        }),
        _error('workspace.work_hours[0]: expected a time'),
      );
      expect(
        _ws({
          'work_hours': ['08:30', '17:00'],
        }).workHours,
        (start: 8 * 60 + 30, end: 17 * 60),
      );
    });

    test('times are 00:00 to 23:59', () {
      for (final bad in ['24:00', '23:60', '25:00', '12:5', 'noon']) {
        expect(
          () => _ws({
            'work_hours': ['00:00', bad],
          }),
          _error('workspace.work_hours[1]: expected a time'),
          reason: bad,
        );
      }
      expect(
        _ws({
          'work_hours': ['00:00', '23:59'],
        }).workHours,
        (start: 0, end: 23 * 60 + 59),
      );
    });

    test('busy spans: start before end, each a pair of times', () {
      expect(
        () => _ws(
          _day({
            'a@x.io': [
              ['11:00', '10:00'],
            ],
          }),
        ),
        _error('workspace.calendars[0].busy.a@x.io[0]: 11:00 is not before 10:00'),
      );
      expect(
        () => _ws(
          _day({
            'a@x.io': [
              ['10:00'],
            ],
          }),
        ),
        _error('workspace.calendars[0].busy.a@x.io[0]: expected ["HH:MM", "HH:MM"]'),
      );
      expect(
        () => _ws(
          _day({
            'a@x.io': [
              ['9', '10:00'],
            ],
          }),
        ),
        _error('workspace.calendars[0].busy.a@x.io[0][0]: expected a time'),
      );
    });

    test('busy: a list of spans or the literal "out", nothing else', () {
      expect(_ws(_day({'a@x.io': 'out'})).calendars.single.busy['a@x.io'], isNull);
      expect(_ws(_day({'a@x.io': []})).calendars.single.clash('a@x.io', 0, 60), isNull);
      for (final bad in ['OOO', true, null, 1, {}]) {
        expect(
          () => _ws(_day({'a@x.io': bad})),
          _error('workspace.calendars[0].busy.a@x.io: expected [["HH:MM", "HH:MM"], …] or "out"'),
          reason: '$bad',
        );
      }
    });

    test('a busy key must be an address, listed once (case-insensitively)', () {
      expect(() => _ws(_day({'Ana Ruiz': 'out'})), _error('busy key: expected an email address, got "ana ruiz"'));
      expect(() => _ws(_day({'a@x.io': 'out', 'A@X.io': 'out'})), _error('busy.a@x.io: listed twice'));
    });
  });

  group('dates', () {
    test('YYYY-MM-DD and a real day', () {
      for (final bad in ['2026-11-3', '03.11.2026', '2026-02-30', '2026-13-01', '2026-11-03T10:00', '']) {
        expect(() => _ws({'today': bad}), _error('workspace.today: expected a date YYYY-MM-DD'), reason: bad);
        expect(
          () => _ws(_day({}, date: bad)),
          _error('workspace.calendars[0].date: expected a date YYYY-MM-DD'),
          reason: bad,
        );
      }
      expect(_ws({'today': '2028-02-29'}).today, DateTime(2028, 2, 29));
    });

    test('one calendar per date', () {
      expect(
        () => _ws({
          'calendars': [
            {'date': '2026-11-18', 'busy': {}},
            {'date': '2026-11-18', 'busy': {}},
          ],
        }),
        _error('workspace.calendars: date 2026-11-18 appears twice'),
      );
    });
  });

  group('people and the internal domain', () {
    test('internal_domain is a bare domain', () {
      expect(() => _ws({'internal_domain': '@x.io'}), _error('workspace.internal_domain: expected a domain'));
      expect(() => _ws({'internal_domain': ''}), _error('workspace.internal_domain'));
      expect(() => _ws({'internal_domain': 'x .io'}), _error('workspace.internal_domain'));
    });

    test('an internal person outside the domain is refused, saying which', () {
      expect(
        () => _ws({
          'people': [
            {'name': 'Ana', 'email': 'ana@x.io'},
            {'name': 'Rick', 'email': 'rick@acme.com'},
          ],
        }),
        _error('workspace.people[1]: rick@acme.com is outside internal_domain "x.io" but internal'),
      );
      expect(
        () => _ws({
          'people': [
            {'name': 'Rick', 'email': 'rick@acme.com', 'internal': true},
          ],
        }),
        _error('people[0]: rick@acme.com is outside'),
      );
    });

    test('an external person inside the domain is refused too: the mail rule would call them internal', () {
      expect(
        () => _ws({
          'people': [
            {'name': 'Ana', 'email': 'ana@x.io', 'internal': false},
          ],
        }),
        _error('workspace.people[0]: ana@x.io is in internal_domain "x.io" but marked "internal": false'),
      );
    });

    test('without a domain, internal is what the file says', () {
      final w = Workspace.fromJson(
        jsonEncode({
          'people': [
            {'name': 'Rick', 'email': 'rick@acme.com'},
            {'name': 'Vendor', 'email': 'v@v.eu', 'internal': false},
          ],
        }),
      );
      expect(w.people.map((p) => p.internal), [true, false]);
    });

    test('emails are addresses, each listed once', () {
      expect(
        () => _ws({
          'people': [
            {'name': 'Ana', 'email': 'ana'},
          ],
        }),
        _error('workspace.people[0].email: expected an email address, got "ana"'),
      );
      expect(
        () => _ws({
          'people': [
            {'name': 'Ana', 'email': 'ana@x.io'},
            {'name': 'Ana R', 'email': 'ANA@x.io'},
          ],
        }),
        _error('workspace.people: email ana@x.io appears twice'),
      );
      expect(
        () => _ws({
          'people': [
            {'name': ' ', 'email': 'ana@x.io'},
          ],
        }),
        _error('workspace.people[0].name: empty'),
      );
    });
  });
}
