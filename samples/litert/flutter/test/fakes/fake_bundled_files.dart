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

import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// [BundledModelFiles] without an app bundle, answering at once (no file
/// I/O, which a widget test's fake clock would never see complete). By
/// default the built-in speech files resolve to `/bundled/<name>` (the fake
/// speech services never open them) and the embedder's lookups fail ("no app
/// bundle in tests"), so the embedder row fails and setup goes on. With
/// [paths], every file resolves (those paths, else `/bundled/<name>`).
final class FakeBundledFiles extends BundledModelFiles {
  FakeBundledFiles({
    this.error = const BundledFileException('No app bundle in tests'),
    this.paths = const {},
    Set<String>? failing,
  }) : failing =
           failing ??
           {kBundledEmbedderModel.asset, kBundledEmbedderTokenizer.asset};

  /// Paths for every built-in file (the embedder found).
  FakeBundledFiles.at(this.paths) : error = null, failing = const {};

  /// The error of a failing lookup.
  final Exception? error;

  /// The asset keys whose lookups fail with [error].
  final Set<String> failing;

  /// Paths by asset key, instead of `/bundled/<name>`.
  final Map<String, String> paths;
  final List<String> requested = [];

  @override
  Future<Result<String>> pathOf(BundledFile file) async {
    requested.add(file.asset);
    if (error case final e? when failing.contains(file.asset)) {
      return Result.error(e);
    }
    return Result.ok(paths[file.asset] ?? '/bundled/${file.name}');
  }
}
