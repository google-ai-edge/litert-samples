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

import 'package:flutter/material.dart';
import 'package:scrollable_positioned_list/scrollable_positioned_list.dart';

import '../../app/demo_config.dart';
import '../../core/engines/backend_catalog.dart';
import '../../core/engines/engine_pool.dart';
import '../../core/transcript/transcript.dart';
import 'long_context_controller.dart';

/// Act 2 — a 45-minute source in one piece. Left: the transcript the
/// audience can read. Right: the model, prefilled before any question,
/// answering with timestamps that jump the viewer to the quoted line.
class Act2Screen extends StatefulWidget {
  const Act2Screen({super.key, required this.config, required this.pool});

  final DemoConfig config;
  final EnginePool pool;

  @override
  State<Act2Screen> createState() => _Act2ScreenState();
}

class _Act2ScreenState extends State<Act2Screen> {
  LongContextController? _ctl;
  Transcript? _transcript;
  String? _loadError;
  final _question = TextEditingController(
    text: 'What cost centre code did finance assign, who said it, and when?',
  );
  final _scroll = ItemScrollController();
  int? _highlighted;

  @override
  void initState() {
    super.initState();
    unawaited(_init());
  }

  Future<void> _init() async {
    try {
      final path = widget.config.act2.transcriptPath;
      final t = path != null
          ? await Transcript.fromFile(path)
          : await Transcript.fromAsset('assets/prompts/meeting_45min.txt');
      if (!mounted) return;
      final ctl = LongContextController(
        pool: widget.pool,
        spec: widget.config.act2,
        fallbackModelPath: widget.config.modelPath,
        transcript: t,
      );
      ctl.addListener(_onCtl);
      setState(() {
        _transcript = t;
        _ctl = ctl;
      });
      await ctl.start();
      if (!mounted) return;
      final auto = widget.config.act2.autorunQuestion;
      if (auto != null && ctl.phase == Act2Phase.ready) {
        await ctl.ask(auto);
        await ctl.runContrast();
        _emitResults(ctl);
      } else if (ctl.phase == Act2Phase.failed && widget.config.autorunExit) {
        _emitResults(ctl);
      }
    } catch (e) {
      if (mounted) setState(() => _loadError = '$e');
    }
  }

  void _emitResults(LongContextController ctl) {
    final json = jsonEncode({'act': 2, 'captured_at': DateTime.now().toIso8601String(), ...ctl.toJson()});
    // ignore: avoid_print
    print('ACT2_JSON $json');
    final p = widget.config.resultsPath;
    if (p != null) File(p).writeAsStringSync('$json\n', mode: FileMode.append);
    if (widget.config.autorunExit) exit(ctl.error == null ? 0 : 1);
  }

  void _onCtl() {
    if (!mounted) return;
    final ctl = _ctl!;
    // Auto-jump to the first citation as it appears while streaming.
    final first = ctl.citations.isEmpty ? null : ctl.citations.first.segment;
    if (first != null && first.index != _highlighted) _jumpTo(first.index);
    setState(() {});
  }

  void _jumpTo(int index) {
    _highlighted = index;
    if (_scroll.isAttached) {
      _scroll.scrollTo(index: index, duration: const Duration(milliseconds: 400), alignment: 0.3);
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final ctl = _ctl;
    final t = _transcript;
    return Scaffold(
      appBar: AppBar(title: const Text('Act 2 — Long context: the whole meeting, in one piece')),
      body: _loadError != null
          ? Center(child: Text('Transcript failed to load: $_loadError'))
          : (ctl == null || t == null)
              ? const Center(child: CircularProgressIndicator())
              : Row(
                  children: [
                    Expanded(flex: 5, child: _TranscriptPane(transcript: t, scroll: _scroll, highlighted: _highlighted)),
                    const VerticalDivider(width: 1),
                    Expanded(flex: 4, child: _ModelPane(ctl: ctl, question: _question, onJump: _jumpTo)),
                  ],
                ),
    );
  }

  @override
  void dispose() {
    _ctl?.removeListener(_onCtl);
    _ctl?.dispose();
    _question.dispose();
    super.dispose();
  }
}

class _TranscriptPane extends StatelessWidget {
  const _TranscriptPane({required this.transcript, required this.scroll, required this.highlighted});
  final Transcript transcript;
  final ItemScrollController scroll;
  final int? highlighted;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Text(
            '${transcript.segments.length} lines · ${transcript.duration.inMinutes} min · ${transcript.source.split(RegExp(r'[\\/]')).last}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
        Expanded(
          child: ScrollablePositionedList.builder(
            itemScrollController: scroll,
            itemCount: transcript.segments.length,
            itemBuilder: (context, i) {
              final s = transcript.segments[i];
              final hot = i == highlighted;
              return Container(
                color: hot ? scheme.tertiaryContainer : null,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                child: RichText(
                  text: TextSpan(
                    style: DefaultTextStyle.of(context).style.copyWith(fontSize: 13),
                    children: [
                      TextSpan(text: '[${s.stamp}] ', style: TextStyle(fontFamily: 'monospace', color: scheme.primary)),
                      TextSpan(text: '${s.speaker}: ', style: const TextStyle(fontWeight: FontWeight.bold)),
                      TextSpan(text: s.text),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

class _ModelPane extends StatelessWidget {
  const _ModelPane({required this.ctl, required this.question, required this.onJump});
  final LongContextController ctl;
  final TextEditingController question;
  final void Function(int index) onJump;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final tokens = ctl.transcriptTokens;
    final budget = tokens == null ? null : tokens / ctl.spec.maxTokens;
    final contrastBudget = tokens == null ? null : tokens / ctl.spec.contrastMaxTokens;
    final ready = ctl.phase == Act2Phase.ready;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('${ctl.modelFileName} · ${backendLabel(ctl.spec.backend)}'
              '${ctl.activeBackend != null ? ' (running on ${backendLabel(ctl.activeBackend)})' : ''}'
              ' · window ${ctl.spec.maxTokens}',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 8),
          _BudgetBar(
            label: 'Context budget — ${ctl.spec.maxTokens} tokens',
            value: budget,
            detail: tokens == null ? 'counting…' : '$tokens tokens in the transcript',
          ),
          const SizedBox(height: 6),
          _BudgetBar(
            label: 'Short-context model — ${ctl.contrastModelFileName}, ${ctl.spec.contrastMaxTokens} tokens',
            value: contrastBudget,
            detail: contrastBudget == null
                ? ''
                : contrastBudget > 1
                    ? 'does not fit (${(contrastBudget * 100).round()} %)'
                    : 'fits',
          ),
          const SizedBox(height: 12),
          _PhaseLine(ctl: ctl),
          const SizedBox(height: 12),
          TextField(
            controller: question,
            enabled: ready,
            onSubmitted: (q) => ctl.ask(q),
            decoration: const InputDecoration(labelText: 'Ask about the meeting', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
            FilledButton.icon(
              onPressed: ready ? () => ctl.ask(question.text) : null,
              icon: const Icon(Icons.send),
              label: const Text('Ask'),
            ),
            OutlinedButton(
              onPressed: ready && !ctl.contrastRunning ? ctl.runContrast : null,
              child: Text(ctl.contrastRunning ? 'Trying ${ctl.spec.contrastMaxTokens}…' : 'Same source into ${ctl.spec.contrastMaxTokens}'),
            ),
            TextButton(onPressed: ready ? ctl.reset : null, child: const Text('Reset context')),
          ]),
          const SizedBox(height: 12),
          if (ctl.citations.isNotEmpty)
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final c in ctl.citations)
                  ActionChip(
                    avatar: const Icon(Icons.link, size: 16),
                    label: Text(c.stamp),
                    onPressed: c.segment == null ? null : () => onJump(c.segment!.index),
                  ),
              ],
            ),
          const SizedBox(height: 8),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: theme.dividerColor),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SingleChildScrollView(
                child: SelectableText(ctl.answer.isEmpty ? (ready ? 'Ask a question.' : '') : ctl.answer),
              ),
            ),
          ),
          if (ctl.answerMetrics != null)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'question TTFT ${ctl.answerTtft?.inMilliseconds} ms · '
                '${ctl.answerMetrics!.tokensPerSecond?.toStringAsFixed(1)} tok/s · '
                '${ctl.answerMetrics!.outputTokens} tokens',
                style: theme.textTheme.bodySmall,
              ),
            ),
          if (ctl.contrastResult != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                '${ctl.contrastModelFileName} at ${ctl.spec.contrastMaxTokens} tokens: ${ctl.contrastResult}',
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
              ),
            ),
          if (ctl.error != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(ctl.error!.split('\n').first, style: TextStyle(color: theme.colorScheme.error)),
            ),
        ],
      ),
    );
  }
}

class _PhaseLine extends StatelessWidget {
  const _PhaseLine({required this.ctl});
  final LongContextController ctl;

  @override
  Widget build(BuildContext context) {
    final text = switch (ctl.phase) {
      Act2Phase.idle => 'Idle',
      Act2Phase.loadingEngine => 'Loading ${ctl.modelFileName} on ${backendLabel(ctl.spec.backend)}…',
      Act2Phase.prefilling => 'Reading the whole transcript into context (${ctl.transcriptTokens ?? '…'} tokens)…',
      Act2Phase.ready => 'Context is warm: load ${ctl.loadTime?.inMilliseconds} ms · '
          'prefill ${ctl.prefillTime?.inSeconds} s'
          '${ctl.prefillMetrics?.inputTokens != null ? ' (${ctl.prefillMetrics!.inputTokens} tokens)' : ''}',
      Act2Phase.answering => 'Answering…',
      Act2Phase.failed => 'Failed',
    };
    final busy = ctl.phase == Act2Phase.loadingEngine || ctl.phase == Act2Phase.prefilling || ctl.phase == Act2Phase.answering;
    return Row(children: [
      if (busy) const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
      if (busy) const SizedBox(width: 8),
      Expanded(child: Text(text)),
    ]);
  }
}

class _BudgetBar extends StatelessWidget {
  const _BudgetBar({required this.label, required this.value, required this.detail});
  final String label;
  final double? value;
  final String detail;

  @override
  Widget build(BuildContext context) {
    final over = (value ?? 0) > 1;
    final scheme = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(children: [
          Expanded(child: Text(label, style: Theme.of(context).textTheme.bodySmall)),
          Text(detail, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: over ? scheme.error : null)),
        ]),
        const SizedBox(height: 2),
        LinearProgressIndicator(
          value: value?.clamp(0, 1),
          color: over ? scheme.error : scheme.primary,
          backgroundColor: scheme.surfaceContainerHighest,
        ),
      ],
    );
  }
}
