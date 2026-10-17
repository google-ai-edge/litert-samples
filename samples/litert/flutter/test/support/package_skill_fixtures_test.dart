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

import 'package_skill_fixtures.dart';

/// The root of the resolved flutter_edge_ai_agent, from the package config
/// `pub get` wrote; null when it cannot be read.
Directory? _agentPackageRoot() {
  final config = File('.dart_tool/package_config.json');
  if (!config.existsSync()) return null;
  final json = jsonDecode(config.readAsStringSync());
  if (json case {'packages': final List<Object?> packages}) {
    for (final package in packages) {
      if (package case {
        'name': 'flutter_edge_ai_agent',
        'rootUri': final String rootUri,
      }) {
        final uri = config.uri.resolve(
          rootUri.endsWith('/') ? rootUri : '$rootUri/',
        );
        return Directory.fromUri(uri);
      }
    }
  }
  return null;
}

void main() {
  test('the get-current-time fixture is the resolved package\'s file, '
      'verbatim (an upgrade that changes it shows here)', () {
    final root = _agentPackageRoot();
    final file = root == null
        ? null
        : File('${root.path}$kPackageSkillsDir/get-current-time/SKILL.md');
    if (file == null || !file.existsSync()) {
      markTestSkipped('flutter_edge_ai_agent sources not found');
      return;
    }
    expect(packageGetCurrentTimeSkillMd, file.readAsStringSync());
  });
}
