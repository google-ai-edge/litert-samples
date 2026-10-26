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

import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';

import '../../app/demo_config.dart';
import '../../core/engines/engine_pool.dart';
import 'meeting_agent_controller.dart';
import 'agent_feed.dart';
import 'checklist_view.dart';
import 'follow_up_board.dart';
import 'meeting_input.dart';
import 'meeting_loader.dart';
import 'stage.dart';
import 'stage_meters.dart';
import 'workspace.dart';

/// Act 4 — a model does the follow-up of a meeting with six tools, and the
/// checklist marks every follow-up and every rule as its calls arrive.
class Act4Screen extends StatefulWidget {
  const Act4Screen({super.key, required this.config, required this.pool});

  final DemoConfig config;
  final EnginePool pool;

  @override
  State<Act4Screen> createState() => _Act4ScreenState();
}

class _Act4ScreenState extends State<Act4Screen> {
  MeetingAgentController? _ctl;
  String? _loadError;

  /// Redraws the clock while a run has nothing new to show (a turn that
  /// writes many calls can stream nothing for over a minute).
  Timer? _tick;

  Act4Spec get spec => widget.config.act4;

  @override
  void initState() {
    super.initState();
    unawaited(_init());
  }

  Future<void> _init() async {
    final MeetingAgentController ctl;
    try {
      final task = spec.meetingPath != null
          ? await MeetingLoader.fromFile(spec.meetingPath!, workspacePath: spec.workspacePath)
          : await MeetingLoader.example(workspacePath: spec.workspacePath);
      if (!mounted) return;
      ctl = MeetingAgentController(pool: widget.pool, spec: spec, task: task)..addListener(_onCtl);
      setState(() => _ctl = ctl);
    } catch (e) {
      if (widget.config.autorunExit) {
        stderr.writeln('ACT4_ERROR $e');
        exit(1);
      }
      if (mounted) setState(() => _loadError = '$e');
      return;
    }
    if (!spec.autorun || !mounted) return;
    var ok = false;
    try {
      ok = await ctl.autorun(widget.config.resultsPath);
    } catch (e, st) {
      stderr.writeln('ACT4_ERROR $e\n$st');
      if (mounted && !widget.config.autorunExit) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('The run failed: $e')));
      }
    } finally {
      if (widget.config.autorunExit) {
        // An NPU bundle's weight bank in the temp folder goes only with its engine.
        await widget.pool.releaseAll();
        exit(ok ? 0 : 1);
      }
    }
  }

  /// The system file dialog; the workspace is the configured one, else the
  /// file beside the transcript.
  Future<void> _openMeeting() async {
    final ctl = _ctl;
    if (ctl == null) return;
    final file = await openFile(
      acceptedTypeGroups: const [XTypeGroup(label: 'Meeting transcripts', extensions: MeetingInput.extensions)],
    );
    if (file == null) return;
    try {
      ctl.openMeeting(await MeetingLoader.fromFile(file.path, workspacePath: spec.workspacePath));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  void _onCtl() {
    if (!mounted) return;
    final ctl = _ctl!;
    final ticking = ctl.phase == Act4Phase.running;
    if (ticking && _tick == null) {
      _tick = Timer.periodic(const Duration(milliseconds: 500), (_) => setState(() {}));
    } else if (!ticking) {
      _tick?.cancel();
      _tick = null;
    }
    setState(() {});
  }

  @override
  void dispose() {
    _tick?.cancel();
    _ctl?.removeListener(_onCtl);
    _ctl?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ctl = _ctl;
    return Theme(
      data: Stage.theme,
      child: Scaffold(
        body: SafeArea(
          child: _loadError != null
              ? Center(child: Text('The task failed to load: $_loadError'))
              : ctl == null
              ? const Center(child: CircularProgressIndicator())
              : spec.models.isEmpty
              ? const Center(child: Text('ACT4_MODELS is not set: no model to run (see README).'))
              : _body(context, ctl),
        ),
      ),
    );
  }

  Widget _body(BuildContext context, MeetingAgentController ctl) {
    final run = ctl.live;
    // Built once per frame: the checklist and the meters read the same board.
    final board = ctl.board;
    final live = ctl.phase == Act4Phase.running;
    final finished = run != null && (run.runS != null || run.error != null) && ctl.phase != Act4Phase.running;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _Header(ctl: ctl, openMeeting: _openMeeting),
          const SizedBox(height: 12),
          StageMeters(items: board, run: run, now: ctl.now, live: live),
          const SizedBox(height: 10),
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  flex: 6,
                  child: Container(
                    decoration: Stage.panelBox(),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(14, 12, 14, 4),
                          child: Row(
                            children: [
                              const Expanded(child: Text('WHAT THE AGENT DID', style: Stage.label)),
                              if (run != null && run.attempts.isNotEmpty)
                                Text(
                                  '${run.calls.length} ACTION${run.calls.length == 1 ? '' : 'S'}',
                                  style: Stage.label.copyWith(color: Stage.mint),
                                ),
                            ],
                          ),
                        ),
                        Expanded(
                          child: run == null || ctl.phase == Act4Phase.loading
                              ? _Idle(loading: ctl.phase == Act4Phase.loading, model: ctl.model)
                              : AgentFeed(run: run, now: ctl.now, live: live, workspace: ctl.task.workspace),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  flex: 5,
                  child: Container(
                    decoration: Stage.panelBox(),
                    child: ChecklistView(
                      items: board,
                      anyAction: FollowUpBoard.actionsOf(run).isNotEmpty,
                      finished: finished,
                      live: live,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.ctl, required this.openMeeting});
  final MeetingAgentController ctl;
  final VoidCallback openMeeting;

  @override
  Widget build(BuildContext context) {
    final idle = !ctl.busy;
    final m = ctl.model;
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'ACT 4 · THE LAPTOP DOES THE MEETING\'S FOLLOW-UP · ${_workspaceLabel(ctl.task.workspace)}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: Stage.label,
              ),
              const SizedBox(height: 3),
              Row(
                children: [
                  Flexible(
                    child: Text(
                      ctl.task.meeting.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontSize: 21, fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(width: 6),
                  IconButton(
                    onPressed: idle ? openMeeting : null,
                    tooltip: 'Open a meeting transcript',
                    icon: const Icon(Icons.folder_open_rounded, color: Stage.dim),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        PopupMenuButton<int>(
          enabled: idle,
          tooltip: 'Model',
          color: Stage.card,
          onSelected: ctl.select,
          itemBuilder: (_) => [
            for (final (i, e) in ctl.spec.models.indexed)
              PopupMenuItem(
                value: i,
                child: Row(children: [Text(e.name), const SizedBox(width: 8), StageTag(backendTag(e))]),
              ),
          ],
          child: _Chip(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Stage.mint,
                  boxShadow: [BoxShadow(color: Stage.mint.withValues(alpha: 0.8), blurRadius: 8)],
                ),
              ),
              const SizedBox(width: 9),
              Text(m.name, style: const TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(width: 8),
              StageTag(backendTag(m)),
              const SizedBox(width: 4),
              const Icon(Icons.arrow_drop_down, color: Stage.dim),
            ],
          ),
        ),
        const SizedBox(width: 8),
        InkWell(
          borderRadius: BorderRadius.circular(999),
          onTap: idle ? () => ctl.set(think: !ctl.think) : null,
          child: _Chip(
            children: [
              Icon(Icons.psychology_alt_outlined, size: 18, color: ctl.think ? Stage.think : Stage.dim),
              const SizedBox(width: 6),
              Text(
                ctl.think ? 'Thinking on' : 'Thinking off',
                style: TextStyle(fontWeight: FontWeight.w600, color: ctl.think ? Stage.text : Stage.dim),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        if (ctl.busy)
          FilledButton.icon(
            onPressed: ctl.phase == Act4Phase.running ? ctl.stop : null,
            style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16)),
            icon: ctl.phase == Act4Phase.loading
                ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.stop_rounded),
            label: Text(ctl.phase == Act4Phase.loading ? 'Loading…' : 'Stop'),
          )
        else
          FilledButton.icon(
            onPressed: ctl.run,
            style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16)),
            icon: const Icon(Icons.play_arrow_rounded),
            label: const Text('Run', style: TextStyle(fontWeight: FontWeight.w700)),
          ),
      ],
    );
  }
}

/// Which workspace the rules are checked against, so a stray workspace.json
/// beside a transcript, or none at all, is visible.
String _workspaceLabel(Workspace w) {
  if (w.isEmpty) return 'NO WORKSPACE';
  final source = w.source;
  if (source == null) return 'EXAMPLE WORKSPACE';
  return 'WORKSPACE ${source.split(RegExp(r'[\\/]')).last.toUpperCase()}';
}

class _Chip extends StatelessWidget {
  const _Chip({required this.children});
  final List<Widget> children;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
    decoration: BoxDecoration(
      color: Stage.card,
      border: Border.all(color: Stage.line),
      borderRadius: BorderRadius.circular(999),
    ),
    child: Row(mainAxisSize: MainAxisSize.min, children: children),
  );
}

/// Before the first run: what is about to happen.
class _Idle extends StatelessWidget {
  const _Idle({required this.loading, required this.model});
  final bool loading;
  final Act4Model model;

  @override
  Widget build(BuildContext context) => Center(
    child: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(loading ? Icons.downloading_rounded : Icons.forum_outlined, size: 40, color: Stage.dim),
          const SizedBox(height: 12),
          Text(
            loading
                ? 'Loading ${model.name} on the ${backendTag(model)}…'
                : 'Press Run: the model reads the meeting, lists what has to happen,\nthen does it with six tools. '
                      'Open another transcript with the folder button.',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Stage.dim, height: 1.4),
          ),
        ],
      ),
    ),
  );
}
