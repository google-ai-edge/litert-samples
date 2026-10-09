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

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

/// Reads system files (`/proc`, `/sys`, `/etc`). Tests pass fake trees.
abstract interface class SystemFiles {
  /// The file's text; null when it is missing or unreadable.
  String? read(String path);

  /// Entry names directly under [dir]; empty when it is missing.
  List<String> list(String dir);
}

/// The real file system.
final class LocalSystemFiles implements SystemFiles {
  const LocalSystemFiles();

  @override
  String? read(String path) {
    try {
      // procfs and sysfs report size 0; readAsBytesSync reads to EOF anyway.
      return utf8.decode(File(path).readAsBytesSync(), allowMalformed: true);
    } on FileSystemException {
      return null;
    }
  }

  @override
  List<String> list(String dir) {
    try {
      return [
        for (final entry in Directory(dir).listSync(followLinks: false))
          // The path, not the URI: sysfs names hold ':' (PCI addresses).
          entry.path.split('/').lastWhere((s) => s.isNotEmpty),
      ]..sort();
    } on FileSystemException {
      return const [];
    }
  }
}

/// What a finished process printed.
final class const ProcessOutput({
  required final int exitCode,
  required final String stdout,

  /// Why a tool failed (`Connection failure: Connection refused`).
  final String stderr = '',
});

/// Runs a helper tool (`vulkaninfo`, `nvpmodel`). Tests pass canned output.
abstract interface class ProcessRunner {
  /// Null when [executable] is not installed, cannot start, or does not
  /// finish within [timeout] (then it is killed). [environment] is added to
  /// the app's (`LC_ALL=C` for output that is parsed).
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout,
    Map<String, String>? environment,
  });
}

/// `Process.start` with a deadline.
final class LocalProcessRunner implements ProcessRunner {
  const LocalProcessRunner();

  @override
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? environment,
  }) async {
    final Process process;
    try {
      process = await Process.start(
        executable,
        arguments,
        environment: environment,
      );
    } on ProcessException {
      return null; // not installed
    }
    // A tool's odd byte must not fail the whole probe.
    final out = process.stdout
        .transform(const Utf8Decoder(allowMalformed: true))
        .join();
    // Read concurrently so a chatty stderr cannot fill the pipe and stall
    // the tool.
    final err = process.stderr
        .transform(const Utf8Decoder(allowMalformed: true))
        .join();
    try {
      final code = await process.exitCode.timeout(timeout);
      return ProcessOutput(
        exitCode: code,
        stdout: await out,
        stderr: await err,
      );
    } on TimeoutException {
      process.kill(ProcessSignal.sigkill);
      debugPrint(
        '[Hardware] $executable ${arguments.join(' ')} did not finish in '
        '${timeout.inSeconds} s; killed',
      );
      // The streams close once the process is gone; nothing to read.
      unawaited(out.then<void>((_) {}, onError: (Object _) {}));
      unawaited(err.then<void>((_) {}, onError: (Object _) {}));
      return null;
    }
  }
}
