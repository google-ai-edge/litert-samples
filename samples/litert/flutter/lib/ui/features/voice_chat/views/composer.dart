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

import 'package:flutter/material.dart';

import 'chat_keys.dart';

/// [leading] (the mic button), a text field, and Send — or Stop while a turn
/// runs.
class Composer extends StatefulWidget {
  const Composer({
    super.key,
    required this.canSend,
    required this.isGenerating,
    required this.onSend,
    required this.onStop,
    this.leading,
  });

  final bool canSend;
  final bool isGenerating;
  final ValueChanged<String> onSend;
  final VoidCallback onStop;
  final Widget? leading;

  @override
  State<Composer> createState() => _ComposerState();
}

class _ComposerState extends State<Composer> {
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focus = FocusNode();

  @override
  void dispose() {
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  void _submit() {
    final text = _controller.text.trim();
    if (!widget.canSend || text.isEmpty) return;
    widget.onSend(text);
    _controller.clear();
    _focus.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: Row(
          spacing: 8,
          children: [
            ?widget.leading,
            Expanded(
              child: TextField(
                key: ChatKeys.input,
                controller: _controller,
                focusNode: _focus,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _submit(),
                decoration: const InputDecoration(
                  hintText: 'Hold the mic and speak, or type',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            if (widget.isGenerating)
              IconButton.filledTonal(
                key: ChatKeys.stop,
                tooltip: 'Stop',
                onPressed: widget.onStop,
                icon: const Icon(Icons.stop),
              )
            else
              IconButton.filled(
                key: ChatKeys.send,
                tooltip: 'Send',
                onPressed: widget.canSend ? _submit : null,
                icon: const Icon(Icons.send),
              ),
          ],
        ),
      ),
    );
  }
}
