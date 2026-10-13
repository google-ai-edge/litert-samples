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

/// The `<file>.sha256` record beside a verified file: `<hex>  <name>` and a
/// newline (the `sha256sum` format), written with `flush: true` once the
/// file is in place, so a later launch trusts the file by its size and
/// record instead of hashing it again. One format for both stores of
/// verified files: the model store's custom chat model (`VerifiedCommit`)
/// and Android's extracted built-in models (`BundledModelFiles`), each with
/// its own commit around it.
abstract final class ChecksumRecord {
  /// The record file of [target].
  static File of(File target) => File('${target.path}.sha256');

  /// The hash [target]'s record names (its first word); null when there is
  /// no record or it is blank.
  static Future<String?> read(File target) async {
    final record = of(target);
    if (!await record.exists()) return null;
    final text = await record.readAsString();
    final first = text.trim().split(RegExp(r'\s+')).first;
    return first.isEmpty ? null : first;
  }

  /// Records [hex] as the SHA-256 of [target] (called [name] in it).
  static Future<void> write(File target, String hex, String name) =>
      of(target).writeAsString('$hex  $name\n', flush: true);
}
