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
import 'dart:collection';

import 'package:flutter/foundation.dart';

/// What the scene tells the narrator: a physics fact in words. Cheap in
/// tokens, no vision, reliable.
class SceneEvent {
  const SceneEvent({
    required this.at,
    required this.kind,
    required this.subject,
    required this.object,
    required this.detail,
  });

  /// Seconds since the scene started.
  final double at;

  /// `collision`, `rest`, `spawn`, `despawn`, `trigger`.
  final String kind;
  final String subject;
  final String object;

  /// Free text the model can quote: speed, height, count.
  final String detail;

  String get stamp {
    final m = (at ~/ 60).toString().padLeft(2, '0');
    final s = (at % 60).toStringAsFixed(1).padLeft(4, '0');
    return '$m:$s';
  }

  String get asText => '[$stamp] $kind: $subject → $object ($detail)';
}

/// Ring buffer + broadcast stream of scene events. The buffer is what the
/// narrator's context log compacts from; the stream is what it batches.
class SceneEventLog extends ChangeNotifier {
  SceneEventLog({this.capacity = 500});

  final int capacity;
  final Queue<SceneEvent> _events = Queue();
  final StreamController<SceneEvent> _stream = StreamController.broadcast();
  int _total = 0;
  final Queue<double> _recentStamps = Queue();

  Stream<SceneEvent> get stream => _stream.stream;
  List<SceneEvent> get recent => List.unmodifiable(_events);
  int get total => _total;

  /// Events in the last second (back-pressure signal for the narrator).
  double get rate {
    if (_recentStamps.isEmpty) return 0;
    final now = _recentStamps.last;
    while (_recentStamps.isNotEmpty && now - _recentStamps.first > 1.0) {
      _recentStamps.removeFirst();
    }
    return _recentStamps.length.toDouble();
  }

  void add(SceneEvent e) {
    _events.addLast(e);
    if (_events.length > capacity) _events.removeFirst();
    _total++;
    _recentStamps.addLast(e.at);
    _stream.add(e);
    notifyListeners();
  }

  void clear() {
    _events.clear();
    _recentStamps.clear();
    _total = 0;
    notifyListeners();
  }

  @override
  void dispose() {
    _stream.close();
    super.dispose();
  }
}
