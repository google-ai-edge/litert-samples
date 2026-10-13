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

import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/detector_worker.dart';
import 'package:litert_edge_demos/data/services/frames/jpeg_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/turbojpeg.dart';
import 'package:litert_edge_demos/utils/worker_channel.dart';

/// What the detector and TurboJPEG services make of their worker's
/// failures and events: the exception texts and log lines callers, the UI
/// and the device logs see. They are the texts each service had before
/// WorkerChannel; the one new line is TurboJPEG's crash.

/// The lines [protocol] logs for [event] with [detail].
List<String> _logged(
  WorkerProtocol protocol,
  WorkerEvent event,
  String detail,
) {
  final lines = <String>[];
  final original = debugPrint;
  debugPrint = (message, {wrapWidth}) => lines.add(message ?? '');
  try {
    protocol.log(event, detail);
  } finally {
    debugPrint = original;
  }
  return lines;
}

void main() {
  group('DetectorService', () {
    const protocol = DetectorService.workerProtocol;

    test(
      'failures are DetectorUnavailableException with the service\'s texts',
      () {
        final texts = {
          for (final kind in WorkerFailure.values)
            kind: protocol.failure(kind, 'the worker exited'),
        };

        expect(texts.values, everyElement(isA<DetectorUnavailableException>()));
        expect(texts.map((kind, e) => MapEntry(kind, '$e')), {
          WorkerFailure.startFailed:
              'Could not start the detector worker: the worker exited',
          WorkerFailure.notRunning:
              'The detector worker is gone: the worker exited',
          WorkerFailure.replyError: 'the worker exited',
          WorkerFailure.lost: 'The detector worker is gone: the worker exited',
        });
      },
    );

    test('log lines', () {
      expect(_logged(protocol, WorkerEvent.unexpectedMessage, 'x'), [
        '[DetectorService] unexpected reply x',
      ]);
      expect(_logged(protocol, WorkerEvent.crashed, 'StateError\n#0 main'), [
        '[DetectorService] worker error: StateError\n#0 main',
      ]);
      expect(_logged(protocol, WorkerEvent.died, 'the worker exited'), [
        '[DetectorService] the worker exited',
      ]);
      expect(_logged(protocol, WorkerEvent.closeFailed, 'gone'), [
        '[DetectorService] close: gone',
      ]);
      expect(_logged(protocol, WorkerEvent.closeTimedOut, '3000 ms'), isEmpty);
    });

    test('the wire format: DetectorReply in, DetectorCloseRequest out; '
        'reasons name "the worker"', () {
      expect((protocol.name, protocol.noun), ('detector', 'the worker'));
      expect(protocol.parseReply(const DetectorReply(3, value: 5)), (
        id: 3,
        value: 5,
        error: null,
      ));
      expect(protocol.parseReply(const DetectorReply(4, error: 'bad')), (
        id: 4,
        value: null,
        error: 'bad',
      ));
      expect(protocol.parseReply('stray'), isNull);
      expect(
        protocol.closeMessage(7),
        isA<DetectorCloseRequest>().having((r) => r.id, 'id', 7),
      );
    });
  });

  group('TurboJpegDecoder', () {
    const protocol = TurboJpegDecoder.workerProtocol;

    test('failures are TurboJpegException with the decoder\'s texts', () {
      final texts = {
        for (final kind in WorkerFailure.values)
          kind: protocol.failure(kind, 'the decoder isolate exited'),
      };

      expect(texts.values, everyElement(isA<TurboJpegException>()));
      expect(texts.map((kind, e) => MapEntry(kind, '$e')), {
        WorkerFailure.startFailed: 'the decoder isolate exited',
        WorkerFailure.notRunning:
            'The decoder is not running (the decoder isolate exited)',
        WorkerFailure.replyError: 'the decoder isolate exited',
        WorkerFailure.lost: 'the decoder isolate exited',
      });
    });

    test('log lines; a crash is logged with its cause (new), a death and a '
        'failed close are not', () {
      expect(_logged(protocol, WorkerEvent.unexpectedMessage, 'x'), [
        '[NetworkCamera] unexpected decoder message: x',
      ]);
      expect(_logged(protocol, WorkerEvent.closeTimedOut, '2000 ms'), [
        '[NetworkCamera] the decoder isolate did not exit: kill',
      ]);
      expect(_logged(protocol, WorkerEvent.crashed, 'StateError\n#0 main'), [
        '[NetworkCamera] decoder isolate error: StateError\n#0 main',
      ]);
      expect(_logged(protocol, WorkerEvent.died, 'exited'), isEmpty);
      expect(_logged(protocol, WorkerEvent.closeFailed, 'gone'), isEmpty);
    });

    test('the wire format: frame records and (id, error) in, null out; '
        'reasons name "the decoder isolate"', () {
      expect(
        (protocol.name, protocol.noun),
        ('turbojpeg', 'the decoder isolate'),
      );
      final rgba = TransferableTypedData.fromList([Uint8List(4)]);
      final frame = (1, 2, 3, 4, 5, rgba, 6);
      expect(protocol.parseReply(frame), (id: 1, value: frame, error: null));
      expect(protocol.parseReply((2, 'Not a JPEG')), (
        id: 2,
        value: null,
        error: 'Not a JPEG',
      ));
      expect(protocol.parseReply('stray'), isNull);
      expect(protocol.parseReply((3, 'too', 'short')), isNull);
      expect(protocol.closeMessage(0), isNull);
    });
  });
}
