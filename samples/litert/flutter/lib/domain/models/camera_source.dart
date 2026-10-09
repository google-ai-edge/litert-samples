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

import '../../utils/result.dart';

/// Where Demo 3's frames come from when the build leaves it to the user
/// (`FRAME_SOURCE` unset or `camera`).
enum CameraSourceKind {
  /// The device's own camera (`camera` plugin).
  device,

  /// An MJPEG stream over HTTP, e.g. a phone running the IP Webcam app on
  /// the same Wi-Fi.
  network;

  /// The persisted name; null for anything else.
  static CameraSourceKind? tryParse(String value) => switch (value) {
    'device' => device,
    'network' => network,
    _ => null,
  };
}

/// The URL the field starts with: IP Webcam's stream path on its default
/// port, with the address left for the user to fill in.
const kNetworkCameraUrlExample = 'http://192.168.x.x:8080/video';

/// The user's camera-source choice (Demo 3's settings). [networkUrl] is kept
/// while the device camera is chosen, so switching back loses nothing.
final class const CameraSourceChoice({
  required final CameraSourceKind kind,
  final String networkUrl = kNetworkCameraUrlExample,
}) {
  /// The device camera with the example URL: the choice before any is saved.
  static const standard = CameraSourceChoice(kind: CameraSourceKind.device);

  CameraSourceChoice copyWith({CameraSourceKind? kind, String? networkUrl}) =>
      CameraSourceChoice(
        kind: kind ?? this.kind,
        networkUrl: networkUrl ?? this.networkUrl,
      );

  @override
  bool operator ==(Object other) =>
      other is CameraSourceChoice &&
      other.kind == kind &&
      other.networkUrl == networkUrl;

  @override
  int get hashCode => Object.hash(kind, networkUrl);
}

/// The URL the user typed is not usable; [message] says how to fix it.
final class InvalidCameraUrlException implements Exception {
  const InvalidCameraUrlException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// [text] as a network camera URL: `http` or `https` with a host. A missing
/// scheme means `http://` (people type `192.168.1.23:8080/video`). The
/// example's `x` placeholders must be replaced.
Result<Uri> parseNetworkCameraUrl(String text) {
  var value = text.trim();
  if (value.isEmpty) {
    return const Result.error(
      InvalidCameraUrlException(
        "Enter the network camera's URL, e.g. http://192.168.1.23:8080/video",
      ),
    );
  }
  if (!value.contains('://')) value = 'http://$value';
  final uri = Uri.tryParse(value);
  if (uri == null || uri.host.isEmpty) {
    // Not echoed: what was typed may hold a password.
    return const Result.error(
      InvalidCameraUrlException(
        'That is not a URL with a host, e.g. http://192.168.1.23:8080/video',
      ),
    );
  }
  if (uri.scheme != 'http' && uri.scheme != 'https') {
    return Result.error(
      InvalidCameraUrlException(
        'A network camera URL starts with http:// (got ${uri.scheme}://)',
      ),
    );
  }
  if (RegExp(r'(^|\.)x(\.|$)').hasMatch(uri.host)) {
    return Result.error(
      InvalidCameraUrlException(
        'Replace ${uri.host} with the camera\'s address (IP Webcam shows it '
        'on the phone after "Start server")',
      ),
    );
  }
  return Result.ok(uri);
}

/// `Network camera · 192.168.1.23:8080`: never the path or any password in
/// the URL.
String networkCameraLabel(Uri url) =>
    'Network camera · ${url.host.contains(':') ? '[${url.host}]' : url.host}'
    ':${url.port}';
