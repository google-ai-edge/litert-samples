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
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../../../domain/models/provisioning.dart';

/// The cancel signal of one model-store operation (a download or an
/// import), shared by the parts that run it: [cancel] stops it,
/// mid-transfer included.
final class CancelToken {
  final Completer<void> _cancelled = Completer<void>();

  /// Interrupts the phase in flight with an error: the request while its
  /// headers are awaited, the body read after that. (`HttpClientRequest
  /// .abort` alone does nothing once the response has arrived.)
  void Function(Object error)? interruptPhase;

  bool get cancelled => _cancelled.isCompleted;
  Future<void> get onCancel => _cancelled.future;

  void interrupt(Object error) => interruptPhase?.call(error);

  void cancel() {
    if (cancelled) return;
    _cancelled.complete();
    interrupt(const OperationCancelledException());
  }
}

/// Where a store operation reports the state of the file it produces:
/// [report] at once (a phase starts or ends), [progress] for bytes moving
/// (the receiver throttles it).
abstract interface class FileStateReporter {
  void report(StoreFileState state);
  void progress(StoreFileState state);
}

/// Deletes [file] when it is there; a failure is logged, not thrown.
Future<void> deleteQuietly(File file) async {
  try {
    if (await file.exists()) await file.delete();
  } on FileSystemException catch (e) {
    debugPrint('[ModelStore] could not delete ${file.path}: $e');
  }
}
