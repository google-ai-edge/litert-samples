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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show TextResult;

import '../../utils/spoken_numbers.dart';
import '../hardware/diagnostics_report.dart' show ModelDiagnostics;
import '../hardware/soc_table.dart';
import '../models/accelerator_evidence.dart';
import '../models/hardware_profile.dart';
import 'app_intent_executor.dart';
import 'app_intents.dart';

/// The facts `device_info` reports; the app builds them from its hardware
/// report at call time.
typedef DeviceFactsSource = String Function();

/// Every intent's handler. Results speak numbers as words (the fp16 GPU
/// path may copy digits wrongly) and are one or two short sentences the
/// model can repeat.
Map<String, AppIntentSpec> buildAppIntents({
  required DeviceFactsSource deviceFacts,
  DateTime Function()? now,
}) {
  final clock = now ?? DateTime.now;
  return {
    AppIntent.deviceInfo: AppIntentSpec(
      handler: (_) => TextResult(deviceFacts()),
    ),
    AppIntent.currentTime: AppIntentSpec(
      handler: (_) => TextResult(describeTime(clock())),
    ),
  };
}

/// `It is 2:03 PM on Thursday, October 2, 2026.`
String describeTime(DateTime t) {
  final hour = t.hour % 12 == 0 ? 12 : t.hour % 12;
  final minute = t.minute.toString().padLeft(2, '0');
  final half = t.hour < 12 ? 'AM' : 'PM';
  return 'It is $hour:$minute $half on ${_weekdays[t.weekday - 1]}, '
      '${_months[t.month - 1]} ${t.day}, ${t.year}.';
}

/// What `device_info` says, and the spoken reply to a live device question
/// (no model rephrasing: the facts are exact). From the hardware report
/// only: the chip (or the phone's SoC), the GPU, and where each loaded model
/// actually runs with how that is known — "confirmed" when its runtime
/// reported the backend (`activeBackend`), "requested" when it cannot.
///
/// "This Mac has an Apple M4 Pro with twenty-four gigabytes of memory and
/// an Apple M4 Pro GPU (Metal). Gemma 4 E2B and the detector run on the GPU
/// (confirmed); speech recognition, speech synthesis and the embedder on
/// the CPU (requested)."
///
/// A phone's chip is its SoC by marketing name where the app's table knows
/// it (`Snapdragon 8 Gen 3 (SM8650)`, from the hardware probe's getprop).
String describeHardware({
  required HardwareProfile? hardware,
  required List<ModelDiagnostics> models,
}) {
  final soc = hardware?.soc;
  final sentences = <String>[];
  if (hardware != null) {
    final chip = soc?.label ?? hardware.cpu.model;
    final gpu = hardware.gpus
        .where((g) => g.kind != GpuKind.virtualDisplay)
        .firstOrNull;
    final memory = hardware.memory == null
        ? ''
        : ' with ${_memoryInWords(hardware.memory!.totalBytes)} of memory';
    final device = switch (hardware.platform) {
      HostPlatform.macos => 'This Mac has',
      HostPlatform.ios || HostPlatform.android => 'This phone has',
      _ => 'This computer has',
    };
    // A name that says GPU itself (`NVIDIA Orin iGPU (Ampere)`, a PCI id's
    // `NVIDIA GPU 0x2b85`) is not called a GPU again.
    final gpuPart = gpu == null || chip == 'unknown'
        ? ''
        : ' and ${_article(gpu.name)} ${gpu.name}'
              '${gpu.name.contains('GPU') ? '' : ' GPU'}'
              '${gpu.api == null ? '' : ' (${gpu.api})'}'
              '${gpu.inferred && hardware.platform == HostPlatform.android ? ' (inferred from the chip)' : ''}';
    if (chip != 'unknown' || gpuPart.isNotEmpty) {
      sentences.add('$device ${_article(chip)} $chip$memory$gpuPart.');
    }
  }
  // Grouped by where they run and how that is known, the chat model first.
  final groups = <String, List<String>>{};
  String? chatGroup;
  for (final m in models) {
    final e = m.evidence;
    if (e == null) continue;
    final where = '${_unit(e.actual, soc)} (${_how(e.backendSource)})';
    final group = groups.putIfAbsent(where, () => []);
    if (m.chat case final chat?) {
      // The chat model leads its group too.
      group.insert(0, chat.name);
      chatGroup = where;
    } else {
      group.add(_spokenName(m.name));
    }
  }
  if (groups.isEmpty) {
    sentences.add('No model is loaded yet.');
    return sentences.join(' ');
  }
  final ordered = [
    if (chatGroup != null) MapEntry(chatGroup, groups[chatGroup]!),
    for (final e in groups.entries)
      if (e.key != chatGroup) e,
  ];
  final clauses = <String>[];
  for (final MapEntry(key: where, value: names) in ordered) {
    final verb = clauses.isEmpty ? (names.length == 1 ? ' runs' : ' run') : '';
    clauses.add('${_listed(names)}$verb on the $where');
  }
  sentences.add('${_capitalized(clauses.join('; '))}.');
  return sentences.join(' ');
}

String _unit(String backend, SocInfo? soc) => switch (backend) {
  'npu' when soc != null && _isQualcomm(soc) => 'Qualcomm NPU',
  'npu' => 'NPU',
  final other => other.toUpperCase(),
};

bool _isQualcomm(SocInfo soc) =>
    soc.manufacturer?.toUpperCase() == 'QTI' ||
    lookupSoc(soc.model)?.vendor == SocVendor.qualcomm;

String _how(EvidenceSource source) => switch (source) {
  EvidenceSource.api => 'confirmed',
  EvidenceSource.log => 'confirmed',
  EvidenceSource.inferred => 'inferred',
  EvidenceSource.requested => 'requested',
};

/// The report's model names, as said aloud.
String _spokenName(String name) => switch (name) {
  final n when n.startsWith('Whisper') => 'speech recognition',
  final n when n.startsWith('moonshine') => 'speech recognition',
  final n when n.startsWith('Inflect') => 'speech synthesis',
  final n when n.startsWith('YOLO26n') => 'the detector',
  final n when n.startsWith('EmbeddingGemma') => 'the embedder',
  final n => n,
};

/// `a` or `an` as [name] is said: by its first letter for a word (`an
/// Apple`, `a Snapdragon`), by that letter's name for an initialism, whose
/// first word is all capitals and digits (`an NVIDIA`, `an MT6989`, `a
/// QEMU`), and by the number for a core count (`a 6×`, `an 8×`, `an 11×`).
String _article(String name) {
  final first = name.split(' ').first;
  final initialism =
      first.length > 1 && RegExp(r'^[A-Z][A-Z0-9]+$').hasMatch(first);
  final vowelSound = switch (first) {
    _ when RegExp(r'^\d').hasMatch(first) => RegExp(
      r'^(8|11(?!\d)|18(?!\d))',
    ).hasMatch(first),
    _ when initialism => RegExp('^[AEFHILMNORSX]').hasMatch(first),
    _ => RegExp('^[AEIOU]', caseSensitive: false).hasMatch(name),
  };
  return vowelSound ? 'an' : 'a';
}

/// `a`, `a and b`, `a, b and c` (duplicates said once).
String _listed(List<String> names) => joinWords(names.toSet().toList());

/// `two point one gigabytes`, `eight hundred fifty megabytes`.
String _memoryInWords(int bytes) {
  const mb = 1 << 20;
  const gb = 1 << 30;
  if (bytes >= gb) {
    final tenths = (bytes / gb * 10).round();
    final whole = numberToWords(tenths ~/ 10);
    return tenths % 10 == 0
        ? '$whole gigabytes'
        : '$whole point ${numberToWords(tenths % 10)} gigabytes';
  }
  return '${numberToWords((bytes / mb).round())} megabytes';
}

String _capitalized(String s) =>
    s.isEmpty ? s : '${s[0].toUpperCase()}${s.substring(1)}';

const _weekdays = [
  'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', //
  'Sunday',
];

const _months = [
  'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', //
  'September', 'October', 'November', 'December',
];
