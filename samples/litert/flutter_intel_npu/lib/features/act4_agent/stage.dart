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

import 'package:flutter/material.dart';

import '../../app/demo_config.dart';
import '../../core/engines/backend_catalog.dart';
import 'workspace.dart';

/// Act 4's stage look, the same as Act 3's meters: dark panels, spaced
/// capital labels, big mint numbers.
abstract final class Stage {
  static const bg = Color(0xFF141519);
  static const panel = Color(0xFF1B1C21);
  static const card = Color(0xFF24262C);
  static const line = Color(0xFF33353D);
  static const track = Color(0xFF3A3C44);
  static const text = Color(0xFFECEDEF);
  static const dim = Color(0xFF9A9CA5);
  static const mint = Color(0xFF5EE6A8);
  static const mintDeep = Color(0xFF1F4D3B);
  static const red = Color(0xFFFF7A7A);
  static const redDeep = Color(0xFF5A2328);
  static const amber = Color(0xFFF5C451);
  static const think = Color(0xFF7C86FF);
  static const thinkDeep = Color(0xFF1E2034);

  static const label = TextStyle(fontSize: 11, letterSpacing: 1.6, fontWeight: FontWeight.w600, color: dim);

  static const tabular = [FontFeature.tabularFigures()];

  /// Built once: a seeded ThemeData computes a whole tonal palette, and the act
  /// rebuilds on every streamed chunk.
  static final ThemeData theme = _theme();

  static ThemeData _theme() {
    final base = ThemeData(brightness: Brightness.dark, useMaterial3: true, colorSchemeSeed: mint);
    return base.copyWith(
      scaffoldBackgroundColor: bg,
      canvasColor: panel,
      colorScheme: base.colorScheme.copyWith(primary: mint, onPrimary: const Color(0xFF0E2A1F), surface: panel),
      dividerColor: line,
      textTheme: base.textTheme.apply(bodyColor: text, displayColor: text),
    );
  }

  static BoxDecoration panelBox({Color color = panel}) =>
      BoxDecoration(color: color, borderRadius: BorderRadius.circular(12));
}

/// Seconds as m:ss once past a minute.
String clock(double s) {
  if (s < 60) return '${s.toStringAsFixed(0)} s';
  final m = s ~/ 60;
  final r = (s - m * 60).floor();
  return '$m:${r.toString().padLeft(2, '0')}';
}

/// A person as the screen names them: the name a directory gives the
/// address, else the address's first part ("jane.doe@example.com" → "Jane").
/// An address the workspace does not call internal keeps its domain, so mail
/// leaving the company shows where it goes: "Jane (example.com)".
String person(Object? v, [Workspace? w]) {
  final s = '$v'.trim();
  final at = s.indexOf('@');
  if (at < 0) return s.isEmpty ? '' : s[0].toUpperCase() + s.substring(1);
  final known = w?.people.where((p) => p.email.toLowerCase() == s.toLowerCase()).firstOrNull;
  final first = s.substring(0, at).split(RegExp(r'[._-]')).first;
  final name = known?.name.split(' ').first ?? (first.isEmpty ? s : first[0].toUpperCase() + first.substring(1));
  return isInternal(s, w) ? name : '$name (${s.substring(at + 1)})';
}

String people(Object? v, [Workspace? w]) => v is List ? v.map((e) => person(e, w)).join(', ') : person(v, w);

/// Only a workspace with a mail domain says what is internal; without one,
/// no address is.
bool isInternal(String address, Workspace? w) => w?.internalDomain != null && w!.isInternal(address);

/// "YYYY-MM-DD" → "Fri 13 Nov"; anything else as given.
String day(Object? v) {
  final d = DateTime.tryParse('$v');
  if (d == null) return '$v';
  const wd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const mo = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return '${wd[d.weekday - 1]} ${d.day} ${mo[d.month - 1]}';
}

/// A small rounded tag: NPU, GPU, CPU, "for approval".
class StageTag extends StatelessWidget {
  const StageTag(this.text, {super.key, this.color = Stage.mint, this.solid = true});
  final String text;
  final Color color;
  final bool solid;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
    decoration: BoxDecoration(
      color: solid ? color : Colors.transparent,
      border: solid ? null : Border.all(color: color),
      borderRadius: BorderRadius.circular(5),
    ),
    child: Text(
      text,
      style: TextStyle(
        fontSize: 10,
        fontWeight: FontWeight.w800,
        letterSpacing: 0.6,
        color: solid ? const Color(0xFF0E2A1F) : color,
      ),
    ),
  );
}

String backendTag(Act4Model m) => backendLabel(m.backend).toUpperCase();
