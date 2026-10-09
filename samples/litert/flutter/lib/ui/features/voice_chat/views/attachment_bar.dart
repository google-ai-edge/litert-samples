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

import '../../../../domain/models/llm_image.dart';
import '../../../core/warning_color.dart';
import 'chat_keys.dart';

/// Above the composer: attach a picture from the gallery or (on phones) the
/// camera; once attached, its thumbnail with a remove button. The picture is
/// sticky: it goes with every question until removed.
class AttachmentBar extends StatelessWidget {
  const AttachmentBar({
    super.key,
    required this.attachment,
    required this.picking,
    required this.showGallery,
    required this.showCamera,
    required this.onGallery,
    required this.onCamera,
    required this.onRemove,
    this.disabledReason,
  });

  final LlmImage? attachment;

  /// A picker is open; the buttons wait for it.
  final bool picking;
  final bool showGallery;
  final bool showCamera;
  final VoidCallback onGallery;
  final VoidCallback onCamera;
  final VoidCallback onRemove;

  /// Photos are off (the chat model has no images): the buttons are
  /// disabled and this is shown instead of the hint.
  final String? disabledReason;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hint = theme.textTheme.labelMedium?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    final image = attachment;
    final off = disabledReason;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 16, 0),
      child: Row(
        spacing: 4,
        children: [
          if (showGallery)
            IconButton(
              key: ChatKeys.attachGallery,
              tooltip: off ?? 'Attach a picture from the gallery',
              onPressed: picking || off != null ? null : onGallery,
              icon: const Icon(Icons.photo_library_outlined),
            ),
          if (showCamera)
            IconButton(
              key: ChatKeys.attachCamera,
              tooltip: off ?? 'Take a photo to ask about',
              onPressed: picking || off != null ? null : onCamera,
              icon: const Icon(Icons.photo_camera_outlined),
            ),
          if (picking)
            const SizedBox.square(
              dimension: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          if (image != null) ...[
            const SizedBox(width: 4),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.memory(
                image.png,
                key: ChatKeys.attachmentThumbnail,
                width: 48,
                height: 48,
                fit: BoxFit.cover,
                cacheHeight: 96,
                gaplessPlayback: true,
                errorBuilder: (context, error, stack) =>
                    const Icon(Icons.broken_image_outlined),
              ),
            ),
            IconButton(
              key: ChatKeys.removeAttachment,
              tooltip: 'Remove the picture',
              visualDensity: VisualDensity.compact,
              onPressed: onRemove,
              icon: const Icon(Icons.close),
            ),
            Expanded(
              child: Text(
                '${image.width}×${image.height} · sent with every question '
                'until removed',
                style: hint,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ] else if (off != null)
            Expanded(
              child: Text(
                off,
                key: ChatKeys.photosOff,
                style: hint?.copyWith(color: kWarningColor),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            )
          else if (!picking)
            Expanded(
              child: Text(
                'Attach a picture to ask about it',
                style: hint,
                overflow: TextOverflow.ellipsis,
              ),
            ),
        ],
      ),
    );
  }
}
