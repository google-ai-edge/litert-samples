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

import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;

import 'meeting_input.dart';
import 'meeting_task.dart';
import 'workspace.dart';

/// Finds a meeting and the workspace that goes with it.
///
/// The workspace is, in order: the one named in the configuration
/// (`ACT4_WORKSPACE`); a file beside the transcript, `<name>.workspace.json`,
/// else `workspace.json`; or none, and the agent knows only the meeting.
/// [Workspace.source] says which file it was (null: none, or the bundled
/// example's).
///
/// Nothing is caught here except to put the file's path in the message: a
/// file that cannot be read or parsed stops the act and says which file.
abstract final class MeetingLoader {
  /// The example the app opens with when no meeting is configured.
  static const exampleMeeting = 'assets/agent/example/nimbus-android-3.0-beta.txt';
  static const exampleWorkspace = 'assets/agent/example/nimbus.workspace.json';

  static Future<MeetingTask> example({String? workspacePath}) async {
    final data = await rootBundle.load(exampleMeeting);
    final meeting = MeetingInput.parse(exampleMeeting, data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
    final workspace = workspacePath != null
        ? await readWorkspace(workspacePath)
        : _parse(exampleWorkspace, await rootBundle.loadString(exampleWorkspace), source: null);
    return MeetingTask(meeting: meeting, workspace: workspace);
  }

  static Future<MeetingTask> fromFile(String path, {String? workspacePath}) async {
    final meeting = await MeetingInput.read(path);
    final ws = workspacePath ?? sidecarOf(path);
    return MeetingTask(meeting: meeting, workspace: ws == null ? Workspace.empty : await readWorkspace(ws));
  }

  /// `<name>.workspace.json`, else `workspace.json`, in the transcript's folder.
  static String? sidecarOf(String transcript) {
    final file = File(transcript);
    final dir = file.parent.path;
    final name = transcript.split(RegExp(r'[\\/]')).last;
    final base = name.contains('.') ? name.substring(0, name.lastIndexOf('.')) : name;
    for (final c in [
      '$dir${Platform.pathSeparator}$base.workspace.json',
      '$dir${Platform.pathSeparator}workspace.json',
    ]) {
      if (File(c).existsSync()) return c;
    }
    return null;
  }

  /// The workspace in [path], with [Workspace.source] set to it.
  /// A [FileSystemException] already names the file; a [FormatException]
  /// is given its path.
  static Future<Workspace> readWorkspace(String path) async =>
      _parse(path, await File(path).readAsString(), source: path);

  static Workspace _parse(String path, String json, {required String? source}) {
    try {
      return Workspace.fromJson(json, source: source);
    } on FormatException catch (e) {
      throw FormatException('$path: ${e.message}');
    }
  }
}
