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

import 'dart:io' show Directory;

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// A model path from a `--dart-define`: absolute as given, otherwise relative
/// to the app's documents directory. Absolute in the running platform's
/// style: `/…` on POSIX; `C:\…`, `C:/…`, `\\server\share\…` or a rooted
/// `\…` on Windows. [style] and [documents] are the tests' seams.
Future<String> resolveLocalPath(
  String path, {
  p.Context? style,
  Future<Directory> Function() documents = getApplicationDocumentsDirectory,
}) async {
  final context = style ?? p.context;
  if (context.isAbsolute(path)) return path;
  return context.join((await documents()).path, path);
}
