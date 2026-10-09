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

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/build_info.dart';

/// `version: "x"` of [package] in pubspec.lock.
String? _lockedVersion(String lock, String package) => RegExp(
  '^  $package:\\n(?:    .*\\n)*?    version: "([^"]+)"',
  multiLine: true,
).firstMatch(lock)?.group(1);

void main() {
  test('kAppVersion is pubspec.yaml\'s version', () {
    final pubspec = File('pubspec.yaml').readAsStringSync();
    final version = RegExp(
      r'^version:\s*(\S+)',
      multiLine: true,
    ).firstMatch(pubspec)!.group(1);
    expect(kAppVersion, version);
  });

  test('kPackageVersions match pubspec.lock (update the map after a pub '
      'upgrade)', () {
    final lock = File('pubspec.lock').readAsStringSync();
    for (final MapEntry(key: package, value: version)
        in kPackageVersions.entries) {
      expect(_lockedVersion(lock, package), version, reason: package);
    }
  });
}
