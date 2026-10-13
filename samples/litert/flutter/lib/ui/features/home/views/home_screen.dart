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

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../../../config/demos.dart';
import '../../../core/debug_overlay.dart';
import '../../../core/device_card.dart';
import '../../../core/warning_color.dart';
import '../view_models/home_view_model.dart';

/// Keys for tests.
abstract final class HomeKeys {
  static ValueKey<String> tile(Demo demo) => ValueKey('home-tile-${demo.name}');
  static const menu = ValueKey('home-menu');
  static const models = ValueKey('home-menu-models');
  static const licenses = ValueKey('home-menu-licenses');
}

/// The launcher: one tile per demo, enabled when its models are ready.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.onOpen, this.onOpenModels});

  /// Opens [Demo]'s screen; completes when that screen is closed. Only
  /// called for an available tile.
  final Future<void> Function(Demo demo) onOpen;

  /// Opens the Models screen (download or import more models); the menu
  /// entry is hidden without it.
  final Future<void> Function()? onOpenModels;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  /// A demo route is open or being pushed: further taps are ignored, so a
  /// fast double tap cannot push two routes (two view models on one chat).
  bool _demoOpen = false;

  Future<void> _open(Demo demo) async {
    if (_demoOpen) return;
    _demoOpen = true;
    try {
      await widget.onOpen(demo);
    } catch (e, st) {
      // A navigation failure is a bug: report it like an uncaught error.
      FlutterError.reportError(
        FlutterErrorDetails(exception: e, stack: st, library: 'home screen'),
      );
    } finally {
      _demoOpen = false;
    }
  }

  /// Like [_open]: no second route while one is open.
  Future<void> _openModels(Future<void> Function() openModels) async {
    if (_demoOpen) return;
    _demoOpen = true;
    try {
      await openModels();
    } catch (e, st) {
      FlutterError.reportError(
        FlutterErrorDetails(exception: e, stack: st, library: 'home screen'),
      );
    } finally {
      _demoOpen = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    final viewModel = context.read<HomeViewModel>();
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('On-device AI demos'),
        actions: [
          const DebugOverlayToggle(),
          if (widget.onOpenModels case final openModels?)
            PopupMenuButton<void>(
              key: HomeKeys.menu,
              tooltip: 'More',
              itemBuilder: (context) => [
                PopupMenuItem<void>(
                  key: HomeKeys.models,
                  onTap: () => unawaited(_openModels(openModels)),
                  child: const Text('Models'),
                ),
                // The built-in models' licences (YOLO26n: AGPL-3.0) with
                // the packages' own.
                PopupMenuItem<void>(
                  key: HomeKeys.licenses,
                  onTap: () => showLicensePage(
                    context: context,
                    applicationName: 'LiteRT Demos',
                  ),
                  child: const Text('Licences'),
                ),
              ],
            ),
        ],
      ),
      body: ListenableBuilder(
        listenable: viewModel,
        builder: (context, _) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            for (final tile in viewModel.tiles)
              _DemoCard(
                tile: tile,
                onTap: tile.available
                    ? () => unawaited(_open(tile.demo))
                    : null,
              ),
            const SizedBox(height: 8),
            Text(
              HomeViewModel.historyNote,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (viewModel.device case final device?) ...[
              const SizedBox(height: 8),
              DeviceCard(summary: device, report: viewModel.diagnosticsReport),
            ],
          ],
        ),
      ),
    );
  }
}

class _DemoCard extends StatelessWidget {
  const _DemoCard({required this.tile, required this.onTap});

  final DemoTile tile;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final demo = tile.demo;
    return Card(
      child: ListTile(
        key: HomeKeys.tile(demo),
        enabled: tile.available,
        onTap: onTap,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Icon(switch (demo) {
          Demo.voiceChat => Icons.forum_outlined,
          Demo.liveCamera => Icons.camera_alt_outlined,
        }, size: 32),
        title: Text(demo.title),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          spacing: 4,
          children: [
            Text(tile.subtitle),
            Text(
              tile.status,
              style: TextStyle(
                color: switch (tile) {
                  DemoTile(warning: true) => kWarningColor,
                  DemoTile(available: true) => colors.primary,
                  _ => colors.onSurfaceVariant,
                },
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        trailing: tile.available ? const Icon(Icons.chevron_right) : null,
      ),
    );
  }
}
