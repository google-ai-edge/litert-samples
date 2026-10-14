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

import 'dart:collection';

import '../../../config/live_camera_config.dart';
import '../../../domain/models/visual_memory.dart';
import '../../../utils/cosine.dart';

/// The remembered frames, oldest first, in memory only: at most
/// [_maxEntries] frames and [_maxBytes] bytes of PNG thumbnails (their
/// measured lengths), the oldest dropped first. Search is a cosine scan
/// (200 × 768 multiply-adds is well under a millisecond).
final class MemoryStore {
  MemoryStore({
    this._maxEntries = kMemoryMaxEntries,
    this._maxBytes = kMemoryMaxBytes,
  });

  final int _maxEntries;
  final int _maxBytes;
  final ListQueue<MemoryEntry> _entries = ListQueue();
  int _bytes = 0;

  int get length => _entries.length;
  bool get isEmpty => _entries.isEmpty;

  /// PNG bytes held.
  int get bytes => _bytes;

  MemoryEntry? get oldest => _entries.isEmpty ? null : _entries.first;
  MemoryEntry? get newest => _entries.isEmpty ? null : _entries.last;

  /// A copy of the frames, oldest first.
  List<MemoryEntry> get entries => List.unmodifiable(_entries);

  /// Adds [entry] as the newest, then drops the oldest frames while either
  /// cap is exceeded (an entry larger than the byte cap on its own is not
  /// kept). Returns how many were dropped.
  int add(MemoryEntry entry) {
    _entries.addLast(entry);
    _bytes += entry.png.length;
    var dropped = 0;
    while (_entries.isNotEmpty &&
        (_entries.length > _maxEntries || _bytes > _maxBytes)) {
      _bytes -= _entries.removeFirst().png.length;
      dropped++;
    }
    return dropped;
  }

  /// The newest frame was seen again, unchanged, at [atMicros]: its time is
  /// refreshed in place, so a static scene's re-store adds no duplicate (a
  /// duplicate would only fill the caps and make "seen N ago" come from
  /// whichever copy won by noise). False when nothing is stored.
  bool refreshNewest(int atMicros) {
    if (_entries.isEmpty) return false;
    _entries.addLast(_entries.removeLast().seenAgainAt(atMicros));
    return true;
  }

  /// [query]'s cosine against every frame, oldest first.
  List<double> cosines(List<double> query) => [
    for (final entry in _entries) cosineSimilarity(query, entry.vector),
  ];

  void clear() {
    _entries.clear();
    _bytes = 0;
  }
}
