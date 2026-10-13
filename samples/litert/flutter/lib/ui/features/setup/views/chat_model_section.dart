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
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;

import '../../../../domain/models/chat_model.dart';
import '../../../../utils/result.dart';
import '../../../core/warning_color.dart';
import '../view_models/chat_model_view_model.dart';

/// Keys for tests.
abstract final class ChatModelKeys {
  static const section = ValueKey('chat-model-section');
  static const note = ValueKey('chat-model-note');
  static const activeLine = ValueKey('chat-model-active');
  static const importFile = ValueKey('chat-model-import');
  static const importUnavailable = ValueKey('chat-model-import-unavailable');
  static const downloadUrl = ValueKey('chat-model-download');
  static const urlField = ValueKey('chat-model-url-field');
  static const shaField = ValueKey('chat-model-sha-field');
  static const sizeField = ValueKey('chat-model-size-field');
  static const urlSubmit = ValueKey('chat-model-url-submit');
  static const urlError = ValueKey('chat-model-url-error');
  static const fileLine = ValueKey('chat-model-file');
  static const replacementPending = ValueKey('chat-model-replacement');
  static const name = ValueKey('chat-model-name');
  static const type = ValueKey('chat-model-type');
  static const backend = ValueKey('chat-model-backend');
  static const npuReason = ValueKey('chat-model-npu-reason');
  static const context = ValueKey('chat-model-context');
  static const images = ValueKey('chat-model-images');
  static const tools = ValueKey('chat-model-tools');
  static const apply = ValueKey('chat-model-apply');
  static const draftProblem = ValueKey('chat-model-draft-problem');
  static const loadError = ValueKey('chat-model-load-error');
  static const cancel = ValueKey('chat-model-cancel');
  static const folderPath = ValueKey('chat-model-folder-path');
  static const copyFolder = ValueKey('chat-model-copy-folder');
  static const rescan = ValueKey('chat-model-rescan');
  static const folderHint = ValueKey('chat-model-folder-hint');
  static const folderEmpty = ValueKey('chat-model-folder-empty');
  static const pathButton = ValueKey('chat-model-path');
  static const pathField = ValueKey('chat-model-path-field');
  static const pathSubmit = ValueKey('chat-model-path-submit');
  static const localError = ValueKey('chat-model-local-error');
  static const localProblem = ValueKey('chat-model-local-problem');

  static ValueKey<String> localFile(String name) =>
      ValueKey('chat-model-local-$name');

  /// The folder at [index] (0: the app's own).
  static ValueKey<String> folder(int index) =>
      ValueKey('chat-model-folder-$index');

  static ValueKey<String> copyFolderAt(int index) =>
      ValueKey('chat-model-copy-folder-$index');

  static ValueKey<String> action(ChatModelAction action) =>
      ValueKey('chat-model-action-${action.name}');
}

/// The Models screen's "Chat model" card: the app ships no chat model, so it
/// leads with how to choose a `.litertlm` (the models folders, Path…, import, a
/// link), then its backend, context, images and tools; a failed load offers
/// "Run on GPU" / "Run on CPU".
class ChatModelSection extends StatelessWidget {
  const ChatModelSection({super.key, required this.viewModel});

  final ChatModelViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) {
        final theme = Theme.of(context);
        final vm = viewModel;
        final caption = theme.textTheme.bodySmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        );
        final error = TextStyle(color: theme.colorScheme.error);
        return Card(
          key: ChatModelKeys.section,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              spacing: 8,
              children: [
                Text('CHAT MODEL', style: theme.textTheme.labelMedium),
                Text(
                  vm.activeLine,
                  key: ChatModelKeys.activeLine,
                  style: theme.textTheme.titleSmall,
                ),
                if (vm.loadError case final message?) ...[
                  Text(message, key: ChatModelKeys.loadError, style: error),
                  if (vm.failureActions.isNotEmpty)
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        for (final action in vm.failureActions)
                          OutlinedButton(
                            key: ChatModelKeys.action(action),
                            onPressed: vm.busy
                                ? null
                                : () => vm.runAction(action),
                            child: Text(switch (action) {
                              ChatModelAction.runOnGpu => 'Run on GPU',
                              ChatModelAction.runOnCpu => 'Run on CPU',
                            }),
                          ),
                      ],
                    ),
                ],
                if (vm.note case final note?)
                  Text(
                    note,
                    key: ChatModelKeys.note,
                    style: caption?.copyWith(color: kWarningColor),
                  ),
                if (vm.noModel)
                  Text(
                    'The app ships no chat model: use a .litertlm such as '
                    'gemma-4-E2B-it.litertlm (GPU) or a Gemma build compiled '
                    "for this phone's Qualcomm NPU.",
                    style: caption,
                  ),
                if (vm.problem case final problem?) Text(problem, style: error),
                if (vm.actionError case final message?)
                  Text(message, style: error),
                _FileRow(viewModel: vm, caption: caption, error: error),
                if (vm.saved != null) _Editor(viewModel: vm),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _FileRow extends StatelessWidget {
  const _FileRow({
    required this.viewModel,
    required this.caption,
    required this.error,
  });

  final ChatModelViewModel viewModel;
  final TextStyle? caption;
  final TextStyle error;

  Future<void> _enterPath(BuildContext context) async {
    final path = await showDialog<String>(
      context: context,
      builder: (_) => const _PathDialog(),
    );
    if (path == null || !context.mounted) return;
    await viewModel.useLocal.execute(path);
  }

  Future<void> _copyFolder(BuildContext context, String path) async {
    await Clipboard.setData(ClipboardData(text: path));
    if (!context.mounted) return;
    ScaffoldMessenger.maybeOf(context)
        ?.showSnackBar(const SnackBar(content: Text('Folder path copied')));
  }

  Widget _folders(BuildContext context) {
    final vm = viewModel;
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 4,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                'Models folders (used in place, no copy)',
                style: theme.textTheme.labelLarge,
              ),
            ),
            IconButton(
              key: ChatModelKeys.rescan,
              tooltip: 'Rescan',
              icon: const Icon(Icons.refresh, size: 20),
              onPressed: vm.rescan.running ? null : vm.rescan.execute,
            ),
          ],
        ),
        if (vm.folders.isEmpty)
          Text('Looking for the folders…', style: caption),
        for (final (index, folder) in vm.folders.indexed)
          Column(
            key: ChatModelKeys.folder(index),
            crossAxisAlignment: CrossAxisAlignment.start,
            spacing: 2,
            children: [
              if (vm.folders.length > 1)
                Text(folder.label, style: theme.textTheme.labelMedium),
              Row(
                children: [
                  Expanded(
                    child: SelectableText(
                      folder.path ?? folder.error ?? '',
                      key: index == 0 ? ChatModelKeys.folderPath : null,
                      style: folder.path == null ? error : caption,
                    ),
                  ),
                  if (folder.path case final path?)
                    IconButton(
                      key: index == 0
                          ? ChatModelKeys.copyFolder
                          : ChatModelKeys.copyFolderAt(index),
                      tooltip: 'Copy the folder path',
                      icon: const Icon(Icons.copy, size: 18),
                      onPressed: () => unawaited(_copyFolder(context, path)),
                    ),
                ],
              ),
              if (folder.hint case final hint?)
                SelectableText(
                  hint,
                  key: index == 0 ? ChatModelKeys.folderHint : null,
                  style: caption,
                ),
              if (folder.error case final message? when folder.path != null)
                Text(message, style: error),
              if (folder.files.isEmpty &&
                  folder.path != null &&
                  folder.error == null)
                Text(
                  'No .litertlm files here yet.',
                  key: index == 0 ? ChatModelKeys.folderEmpty : null,
                  style: caption,
                )
              else
                for (final entry in folder.files) _localTile(entry),
            ],
          ),
        TextButton.icon(
          key: ChatModelKeys.pathButton,
          onPressed: vm.busy ? null : () => unawaited(_enterPath(context)),
          icon: const Icon(Icons.edit_location_alt_outlined, size: 18),
          label: const Text('Path…'),
        ),
        if (vm.localError case final message?)
          Text(message, key: ChatModelKeys.localError, style: error),
        if (vm.localProblem case final message?)
          Text(message, key: ChatModelKeys.localProblem, style: error),
      ],
    );
  }

  Widget _localTile(LocalModelEntry entry) {
    final vm = viewModel;
    final selected = vm.isSelected(entry);
    return ListTile(
      key: ChatModelKeys.localFile(entry.name),
      dense: true,
      contentPadding: EdgeInsets.zero,
      leading: Icon(
        selected ? Icons.radio_button_checked : Icons.radio_button_unchecked,
        size: 20,
      ),
      title: Text(entry.name),
      subtitle: Text(ChatModelViewModel.entryLine(entry)),
      selected: selected,
      enabled: !vm.busy,
      onTap: vm.canPick(entry)
          ? () => unawaited(vm.useLocal.execute(entry.path))
          : null,
    );
  }

  Future<void> _downloadFromUrl(BuildContext context) async {
    final request = await showDialog<ModelUrlRequest>(
      context: context,
      builder: (_) => const _UrlDialog(),
    );
    if (request == null || !context.mounted) return;
    await viewModel.download.execute(request);
  }

  @override
  Widget build(BuildContext context) {
    final vm = viewModel;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 8,
      children: [
        if (vm.fileLine case final line?)
          Text(line, key: ChatModelKeys.fileLine, style: caption)
        else
          Text('No file yet.', style: caption),
        if (vm.replacementPending)
          Text(
            'This file is saved, but the previous one still runs: press '
            '"${vm.applyLabel}" to load it.',
            key: ChatModelKeys.replacementPending,
            style: caption?.copyWith(color: kWarningColor),
          ),
        if (vm.fileProgress case (final label, final fraction)) ...[
          Text(label),
          LinearProgressIndicator(value: fraction),
        ],
        _folders(context),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            if (vm.importUnavailableReason == null)
              OutlinedButton(
                key: ChatModelKeys.importFile,
                onPressed: vm.canImport ? vm.importFile.execute : null,
                child: const Text('Import file…'),
              ),
            OutlinedButton(
              key: ChatModelKeys.downloadUrl,
              onPressed: vm.canDownload
                  ? () => unawaited(_downloadFromUrl(context))
                  : null,
              child: const Text('Download from URL…'),
            ),
            if (vm.fileProgress != null)
              TextButton(
                key: ChatModelKeys.cancel,
                onPressed: vm.cancel,
                child: const Text('Cancel'),
              ),
          ],
        ),
        if (vm.importUnavailableReason case final reason?)
          Text(reason, key: ChatModelKeys.importUnavailable, style: caption),
        if (vm.transferError case final message?) Text(message, style: error),
        if (vm.importUnavailableReason == null)
          Text(
            'Import copies the file into the app (instantly on APFS) and '
            'records its SHA-256; a file in a models folder is used where it '
            'is.',
            style: caption,
          ),
      ],
    );
  }
}

/// The custom model's settings. Owns the text controllers.
class _Editor extends StatefulWidget {
  const _Editor({required this.viewModel});

  final ChatModelViewModel viewModel;

  @override
  State<_Editor> createState() => _EditorState();
}

class _EditorState extends State<_Editor> {
  late final TextEditingController _name = TextEditingController(
    text: widget.viewModel.draftName,
  );
  late final TextEditingController _context = TextEditingController(
    text: widget.viewModel.draftContext,
  );

  @override
  void didUpdateWidget(_Editor oldWidget) {
    super.didUpdateWidget(oldWidget);
    _sync(_name, widget.viewModel.draftName);
    _sync(_context, widget.viewModel.draftContext);
  }

  /// A new file reset the draft: the fields follow it.
  static void _sync(TextEditingController controller, String value) {
    if (controller.text != value) controller.text = value;
  }

  @override
  void dispose() {
    _name.dispose();
    _context.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final vm = widget.viewModel;
    final theme = Theme.of(context);
    final caption = theme.textTheme.bodySmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    final hint = vm.contextHint;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 8,
      children: [
        TextField(
          key: ChatModelKeys.name,
          controller: _name,
          decoration: const InputDecoration(labelText: 'Name'),
          onChanged: vm.setName,
        ),
        Row(
          spacing: 12,
          children: [
            const Text('Model type'),
            DropdownButton<ModelType>(
              key: ChatModelKeys.type,
              value: vm.draftType,
              onChanged: (type) {
                if (type != null) vm.setType(type);
              },
              items: [
                for (final type in kCustomModelTypes)
                  DropdownMenuItem(value: type, child: Text(type.name)),
              ],
            ),
          ],
        ),
        Text(modelTypeNote(vm.draftType), style: caption),
        SegmentedButton<PreferredBackend>(
          key: ChatModelKeys.backend,
          segments: [
            ButtonSegment(
              value: PreferredBackend.npu,
              label: const Text('NPU'),
              enabled: vm.npuOffered,
              tooltip: vm.npuUnavailableReason,
            ),
            const ButtonSegment(
              value: PreferredBackend.gpu,
              label: Text('GPU'),
            ),
            const ButtonSegment(
              value: PreferredBackend.cpu,
              label: Text('CPU'),
            ),
          ],
          selected: {vm.draftBackend},
          onSelectionChanged: (selection) => vm.setBackend(selection.single),
        ),
        Text(
          vm.npuUnavailableReason ?? 'NPU ${vm.npuLine}',
          key: ChatModelKeys.npuReason,
          style: vm.npuOffered
              ? caption
              : caption?.copyWith(color: kWarningColor),
        ),
        TextField(
          key: ChatModelKeys.context,
          controller: _context,
          keyboardType: TextInputType.number,
          decoration: InputDecoration(
            labelText: 'Context length (tokens)',
            helperText: [
              'NPU builds have one compiled length: enter that one',
              if (hint != null) 'the file name says $hint',
            ].join('; '),
            helperMaxLines: 2,
          ),
          onChanged: vm.setContext,
        ),
        SwitchListTile(
          key: ChatModelKeys.images,
          contentPadding: EdgeInsets.zero,
          title: const Text('Images'),
          subtitle: const Text(
            'Load the vision encoder. Off: photos in Voice chat and detailed '
            'answers in Live camera are disabled.',
          ),
          value: vm.draftImages,
          onChanged: (value) => vm.setImages(enabled: value),
        ),
        SwitchListTile(
          key: ChatModelKeys.tools,
          contentPadding: EdgeInsets.zero,
          title: const Text('Tools'),
          subtitle: const Text(
            'Send tool declarations (skills). Off: skills that need tool '
            'calls are disabled.',
          ),
          value: vm.draftTools,
          onChanged: (value) => vm.setTools(enabled: value),
        ),
        if (vm.draftProblem case final problem?)
          Text(
            problem,
            key: ChatModelKeys.draftProblem,
            style: TextStyle(color: theme.colorScheme.error),
          ),
        Align(
          alignment: Alignment.centerRight,
          child: FilledButton(
            key: ChatModelKeys.apply,
            onPressed: vm.canApply ? vm.apply.execute : null,
            child: Text(vm.applyLabel),
          ),
        ),
      ],
    );
  }
}

/// Asks for the link, an optional SHA-256 and an optional size; returns the
/// checked request, or null on cancel.
class _UrlDialog extends StatefulWidget {
  const _UrlDialog();

  @override
  State<_UrlDialog> createState() => _UrlDialogState();
}

class _UrlDialogState extends State<_UrlDialog> {
  final _url = TextEditingController();
  final _sha = TextEditingController();
  final _size = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _url.dispose();
    _sha.dispose();
    _size.dispose();
    super.dispose();
  }

  void _submit() {
    switch (ModelUrlRequest.parse(_url.text, _sha.text, _size.text)) {
      case Ok(:final value):
        Navigator.of(context).pop(value);
      case Error(:final error):
        setState(
          () => _error = error is FormatException ? error.message : '$error',
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Download your .litertlm'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              key: ChatModelKeys.urlField,
              controller: _url,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'Link (https://…)',
                hintText: 'Hugging Face resolve/… or a Drive direct link',
              ),
            ),
            TextField(
              key: ChatModelKeys.shaField,
              controller: _sha,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'SHA-256 (optional)',
                helperText: 'Checked when given; otherwise computed',
              ),
            ),
            TextField(
              key: ChatModelKeys.sizeField,
              controller: _size,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Size in bytes (optional)',
                helperText: 'Needed only if the server does not report it',
              ),
            ),
            if (_error case final error?)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  error,
                  key: ChatModelKeys.urlError,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          key: ChatModelKeys.urlSubmit,
          onPressed: _submit,
          child: const Text('Download'),
        ),
      ],
    );
  }
}

/// "Path…": the absolute path of any `.litertlm`, used in place.
class _PathDialog extends StatefulWidget {
  const _PathDialog();

  @override
  State<_PathDialog> createState() => _PathDialogState();
}

class _PathDialogState extends State<_PathDialog> {
  final TextEditingController _path = TextEditingController();

  @override
  void dispose() {
    _path.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('Use a .litertlm in place'),
    // Wrapped, not scrolled: a long path is read whole before Use.
    content: TextField(
      key: ChatModelKeys.pathField,
      controller: _path,
      autofocus: true,
      minLines: 1,
      maxLines: 4,
      keyboardType: TextInputType.multiline,
      textInputAction: TextInputAction.done,
      decoration: const InputDecoration(
        labelText: 'Absolute path of the file',
        hintText: '/data/local/tmp/litert-models/model.litertlm',
        hintMaxLines: 2,
      ),
      onSubmitted: (value) => Navigator.of(context).pop(value.trim()),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.of(context).pop(),
        child: const Text('Cancel'),
      ),
      FilledButton(
        key: ChatModelKeys.pathSubmit,
        onPressed: () => Navigator.of(context).pop(_path.text.trim()),
        child: const Text('Use'),
      ),
    ],
  );
}
