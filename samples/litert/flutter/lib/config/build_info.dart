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

import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';

import '../domain/models/hardware_profile.dart' show BuildInfo;

/// `version:` in pubspec.yaml. test/config/build_info_test.dart keeps it
/// honest (no package_info_plus for one string).
const kAppVersion = '1.0.0+1';

/// Resolved versions of the packages that decide where the models run, as in
/// pubspec.lock (checked by test/config/build_info_test.dart: a `pub upgrade`
/// fails that test until this map is updated).
const kPackageVersions = {
  'flutter_edge_ai': '2.1.1',
  'flutter_edge_ai_litertlm': '1.11.0',
  'flutter_litert': '3.9.3',
};

/// Set by the flutter tool for every build (`flutter_command.dart`
/// `flutterVersionDefine`); empty when a build bypasses it.
const kFlutterVersion = String.fromEnvironment('FLUTTER_VERSION');

/// What this binary is: versions and build mode, for the diagnostics report,
/// the startup line and the self-test.
BuildInfo currentBuildInfo() => BuildInfo(
  appVersion: kAppVersion,
  buildMode: kReleaseMode
      ? 'release'
      : kProfileMode
      ? 'profile'
      : 'debug',
  flutterVersion: kFlutterVersion.isEmpty ? null : kFlutterVersion,
  dartVersion: Platform.version.split(' ').first,
  packages: kPackageVersions,
);
