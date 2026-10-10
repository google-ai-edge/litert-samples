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

// tool/fetch_models.sh against a stub curl that serves files from a local
// folder (no network), honours -C - and -L, and logs each request with the
// Authorization it got and whether -q came first: verify-only mode, the
// gated-repo token (environment or .env, never printed, sent only for gated
// files), idempotent reruns, a resumed download, a redirect, a corrupt
// local file, a corrupt or refused download, and the YOLO26n venv's setup
// against a stub python3. The real derivation needs Python and Arm's
// original; the clean-checkout run covers it.
@TestOn('linux || mac-os')
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';

const _script = 'tool/fetch_models.sh';
const _revision = '0123456789abcdef0123456789abcdef01234567';
const _token = 'hf_secret_token_42';

const _curl = r'''#!/bin/sh
# Stub curl: serves $REMOTE/<url path> into -o, or exits 22 with the HTTP
# code in <file>.status. <file>.redirect names the path it redirects to,
# followed only with -L (without it: an empty body, 302, exit 0, like curl
# without -f failing on a 3xx). With `-C -` it resumes: only the bytes past
# the output file's length are appended. Logs "<url> <auth>" to $CURL_LOG,
# plus " resume=<offset>" for a resumed download, " via=<path>" for a
# followed redirect and " NO-Q" when -q is not the first argument (curl then
# reads ~/.curlrc).
noq=' NO-Q'
[ "${1:-}" = -q ] && noq=
out='' url='' config=0 resume=0 follow=0
while [ $# -gt 0 ]; do
  case $1 in
    -o) out=$2; shift ;;
    -w | --retry) shift ;;
    -C) [ "$2" = - ] && resume=1; shift ;;
    -K) [ "$2" = - ] && config=1; shift ;;
    --location) follow=1 ;;
    https://*) url=$1 ;;
    -[!-]*) case $1 in *L*) follow=1 ;; esac ;;
  esac
  shift
done
auth=none
if [ "$config" = 1 ]; then
  token=$(sed -n 's/^header = "Authorization: Bearer \(.*\)"$/\1/p')
  auth="bearer:$token"
fi
path=${url#https://huggingface.co/}
via=
if [ -f "$REMOTE/$path.redirect" ]; then
  if [ "$follow" = 0 ]; then
    echo "$url $auth$noq" >>"$CURL_LOG"
    : >"$out"
    printf 302
    exit 0
  fi
  path=$(cat "$REMOTE/$path.redirect")
  via=" via=$path"
fi
src="$REMOTE/$path"
have=0
[ "$resume" = 1 ] && [ -f "$out" ] && have=$(wc -c <"$out" | tr -d ' ')
from=
[ "$have" -gt 0 ] && from=" resume=$have"
echo "$url $auth$from$via$noq" >>"$CURL_LOG"
if [ -f "$src.status" ]; then cat "$src.status"; exit 22; fi
[ -f "$src" ] || { printf 404; exit 22; }
if [ "$have" -gt 0 ]; then
  tail -c +"$((have + 1))" "$src" >>"$out"
else
  cat "$src" >"$out"
fi
printf 200
''';

/// Stub python3: only `-m venv [--clear] <dir>`. The venv gets the stub
/// [_venvPython] and pip, unless $VENV_NO_PIP is set: that plays Debian
/// without python3-venv, where ensurepip fails after bin/python is made.
const _python3 = r'''#!/bin/sh
[ "$1" = -m ] && [ "$2" = venv ] || exit 2
shift 2
clear=
[ "$1" = --clear ] && clear=' --clear' && shift
echo "venv$clear" >>"$PY_LOG"
[ -n "$clear" ] && rm -f "$1/pip" "$1/.requirements"
mkdir -p "$1/bin"
cp "$VENV_PYTHON" "$1/bin/python"
chmod +x "$1/bin/python"
if [ -n "${VENV_NO_PIP:-}" ]; then
  echo 'Error: ensurepip is not available' >&2
  exit 1
fi
: >"$1/pip"
''';

/// Stub venv python: `-m pip …` works only when the venv has pip;
/// `-I <script> <src> <out>` (the pruning script) copies `<src>` to `<out>`.
const _venvPython = r'''#!/bin/sh
venv=$(cd "$(dirname "$0")/.." && pwd)
case $1 in
  -m)
    echo "pip $3" >>"$PY_LOG"
    [ -f "$venv/pip" ] || { echo 'No module named pip' >&2; exit 1; }
    ;;
  -I) cp "$3" "$4" ;;
  *) exit 2 ;;
esac
''';

String _sha(List<int> bytes) => sha256.convert(bytes).toString();

final class _Checkout {
  _Checkout(this.root, this.shell)
    : repo = Directory('${root.path}/repo'),
      remote = Directory('${root.path}/remote'),
      bin = Directory('${root.path}/bin') {
    Directory('${repo.path}/tool').createSync(recursive: true);
    bin.createSync();
    File(_script).copySync('${repo.path}/$_script');
    File('${bin.path}/curl').writeAsStringSync(_curl);
    File('${bin.path}/python3').writeAsStringSync(_python3);
    File('${root.path}/venv_python').writeAsStringSync(_venvPython);
    Process.runSync('chmod', ['+x', '${bin.path}/curl', '${bin.path}/python3']);
    serve('owner/pub', 'files/a.bin', publicBytes);
    serve('owner/gated', 'b.bin', gatedBytes);
    File('${repo.path}/tool/models.lock').writeAsStringSync('''
# test lock
pip ai-edge-litert==2.2.0

hf a.bin ${publicBytes.length} ${_sha(publicBytes)} public owner/pub $_revision files/a.bin
hf sub/b.bin ${gatedBytes.length} ${_sha(gatedBytes)} gated owner/gated $_revision b.bin
''');
  }

  final Directory root;
  final String shell;
  final Directory repo;
  final Directory remote;
  final Directory bin;
  final publicBytes = utf8.encode('public model bytes');
  final gatedBytes = utf8.encode('gated model bytes!');

  File get log => File('${root.path}/curl.log');
  List<String> get requests => log.existsSync() ? log.readAsLinesSync() : [];
  File get pythonLog => File('${root.path}/python.log');
  List<String> get pythonCalls =>
      pythonLog.existsSync() ? pythonLog.readAsLinesSync() : [];

  /// Replaces the lock.
  void lock(String text) =>
      File('${repo.path}/tool/models.lock').writeAsStringSync(text);
  File model(String dest) => File('${repo.path}/assets/models/$dest');

  File serve(String repoId, String path, List<int> bytes) =>
      File('${remote.path}/$repoId/resolve/$_revision/$path')
        ..createSync(recursive: true)
        ..writeAsBytesSync(bytes);

  Future<ProcessResult> run(
    List<String> args, {
    Map<String, String> env = const {},
  }) => Process.run(
    shell,
    ['${repo.path}/$_script', ...args],
    environment: {
      'PATH': '${bin.path}:/usr/bin:/bin:/usr/sbin:/sbin',
      'HOME': root.path,
      'REMOTE': remote.path,
      'CURL_LOG': log.path,
      'PY_LOG': pythonLog.path,
      'VENV_PYTHON': '${root.path}/venv_python',
      ...env,
    },
    includeParentEnvironment: false,
  );
}

String _output(ProcessResult r) => '${r.stdout}${r.stderr}';

/// /bin/sh (bash on macOS, often dash on Linux) and dash when installed: the
/// script is POSIX sh.
final _shells = ['/bin/sh', if (File('/bin/dash').existsSync()) '/bin/dash'];

void main() {
  test('shellcheck is clean (when installed)', () {
    final which = Process.runSync('/bin/sh', ['-c', 'command -v shellcheck']);
    final shellcheck = (which.stdout as String).trim();
    if (shellcheck.isEmpty) {
      markTestSkipped('shellcheck not installed; the runs below still run');
      return;
    }
    final result = Process.runSync(shellcheck, ['-s', 'sh', _script]);
    expect(result.exitCode, 0, reason: _output(result));
  });

  for (final shell in _shells) {
    group(shell, () {
      late Directory temp;
      late _Checkout checkout;

      setUp(() {
        temp = Directory.systemTemp.createTempSync('fetch_models_test');
        checkout = _Checkout(temp, shell);
      });
      tearDown(() => temp.deleteSync(recursive: true));

      test('--check without the files fails, names each one and says how to '
          'fetch them; no network', () async {
        final result = await checkout.run(const ['--check']);
        expect(result.exitCode, 1, reason: _output(result));
        expect(result.stderr, contains('MISSING     a.bin\n'));
        expect(result.stderr, contains('MISSING     sub/b.bin\n'));
        expect(
          result.stderr,
          contains('2 of 2 model files missing or different'),
        );
        expect(result.stderr, contains('run tool/fetch_models.sh'));
        expect(checkout.requests, isEmpty);
      });

      test('a gated file and no HF_TOKEN: fails before downloading anything, '
          'with the repo page to accept the terms on', () async {
        final result = await checkout.run(const []);
        expect(result.exitCode, 1, reason: _output(result));
        expect(
          result.stderr,
          contains(
            'HF_TOKEN is not set, and these files come from a gated '
            'repo: sub/b.bin',
          ),
        );
        expect(result.stderr, contains('https://huggingface.co/owner/gated'));
        expect(checkout.requests, isEmpty);
        expect(checkout.model('a.bin').existsSync(), isFalse);
      });

      test('HF_TOKEN from the environment: fetches both, sends the token only '
          'for the gated file and never prints it; a rerun downloads '
          'nothing', () async {
        final first = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(first.exitCode, 0, reason: _output(first));
        expect(checkout.model('a.bin').readAsBytesSync(), checkout.publicBytes);
        expect(
          checkout.model('sub/b.bin').readAsBytesSync(),
          checkout.gatedBytes,
        );
        expect(checkout.requests, [
          'https://huggingface.co/owner/pub/resolve/$_revision/files/a.bin none',
          'https://huggingface.co/owner/gated/resolve/$_revision/b.bin '
              'bearer:$_token',
        ]);
        expect(_output(first), isNot(contains(_token)));
        expect(
          first.stdout,
          contains('0 already verified, 2 fetched, 0 failed'),
        );

        final second = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(second.exitCode, 0, reason: _output(second));
        expect(second.stdout, contains('  ok          a.bin'));
        expect(
          second.stdout,
          contains('2 already verified, 0 fetched, 0 failed'),
        );
        expect(checkout.requests, hasLength(2), reason: 'no new request');

        final check = await checkout.run(const ['--check']);
        expect(check.exitCode, 0, reason: _output(check));
        expect(check.stdout, contains('models verified: 2 files'));
      });

      test(
        'HF_TOKEN from .env (export, quotes, CRLF) and not printed',
        () async {
          File('${checkout.repo.path}/.env').writeAsStringSync(
            'OTHER=1\r\nexport HF_TOKEN="$_token"\r\nVOICE_GATE_DBFS=-40\r\n',
          );
          final result = await checkout.run(const []);
          expect(result.exitCode, 0, reason: _output(result));
          expect(checkout.requests.last, endsWith('bearer:$_token'));
          expect(_output(result), isNot(contains(_token)));
        },
      );

      for (final (name, line) in [
        ('a comment after the quotes', 'HF_TOKEN="$_token"  # read token'),
        ('trailing blanks', 'HF_TOKEN=$_token \t '),
        ('single quotes, a comment, CRLF', "HF_TOKEN='$_token' #x\r"),
      ]) {
        test('HF_TOKEN from .env with $name: the bare token is sent', () async {
          File('${checkout.repo.path}/.env').writeAsStringSync('$line\n');
          final result = await checkout.run(const []);
          expect(result.exitCode, 0, reason: _output(result));
          expect(checkout.requests.last, endsWith(' bearer:$_token'));
          expect(_output(result), isNot(contains(_token)));
        });
      }

      test('a local file with the right size but other bytes: --check says the '
          'SHA-256 differs, a fetch replaces it', () async {
        checkout.model('a.bin')
          ..createSync(recursive: true)
          ..writeAsBytesSync(List.filled(checkout.publicBytes.length, 0x2a));
        checkout.model('sub/b.bin')
          ..createSync(recursive: true)
          ..writeAsBytesSync(checkout.gatedBytes);

        final check = await checkout.run(const ['--check']);
        expect(check.exitCode, 1, reason: _output(check));
        expect(
          check.stderr,
          contains('DIFFERENT   a.bin (SHA-256 differs from the lock)'),
        );

        final fetch = await checkout.run(const []);
        expect(fetch.exitCode, 0, reason: _output(fetch));
        expect(checkout.model('a.bin').readAsBytesSync(), checkout.publicBytes);
        expect(checkout.requests, hasLength(1), reason: 'b.bin was verified');
      });

      test('a download with other bytes fails that file, keeps it out of '
          'assets/models and fetches the rest', () async {
        checkout.serve('owner/pub', 'files/a.bin', utf8.encode('tampered'));
        final result = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(result.exitCode, 1, reason: _output(result));
        expect(
          result.stderr,
          contains(
            'FAILED      a.bin: downloaded owner/pub/files/a.bin at $_revision: '
            '8 bytes, want ${checkout.publicBytes.length}',
          ),
        );
        expect(checkout.model('a.bin').existsSync(), isFalse);
        expect(
          checkout.model('sub/b.bin').readAsBytesSync(),
          checkout.gatedBytes,
        );
        expect(
          result.stdout,
          contains('0 already verified, 1 fetched, 1 failed'),
        );
      });

      test('YOLO26n: a venv left without pip by a failed setup is made '
          'again, not reused; a working one is reused', () async {
        final original = utf8.encode('arm yolo26n original');
        checkout
          ..serve('owner/arm', 'orig.tflite', original)
          ..lock('''
pip ai-edge-litert==2.2.0
yolo26n-rawhead y.tflite ${original.length} ${_sha(original)} public owner/arm $_revision orig.tflite ${original.length} ${_sha(original)}
''');

        final broken = await checkout.run(const [], env: {'VENV_NO_PIP': '1'});
        expect(broken.exitCode, 1, reason: _output(broken));
        expect(broken.stderr, contains('-m venv failed'));

        final repaired = await checkout.run(const []);
        expect(repaired.exitCode, 0, reason: _output(repaired));
        expect(repaired.stdout, contains('has no pip'));
        expect(checkout.model('y.tflite').readAsBytesSync(), original);
        expect(checkout.pythonCalls, [
          'venv',
          'pip --version',
          'venv --clear',
          'pip install',
        ]);

        checkout.model('y.tflite').deleteSync();
        final again = await checkout.run(const []);
        expect(again.exitCode, 0, reason: _output(again));
        expect(checkout.pythonCalls.skip(4), [
          'pip --version',
        ], reason: 'the working venv is reused, nothing reinstalled');
        expect(checkout.requests, hasLength(1), reason: 'the original kept');
      });

      test('a partial download is resumed from where it stopped', () async {
        final part = File(
          '${checkout.repo.path}/build/fetch_models/downloads/'
          '${_sha(checkout.publicBytes)}.part',
        )..createSync(recursive: true);
        part.writeAsBytesSync(checkout.publicBytes.sublist(0, 7));

        final result = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(result.exitCode, 0, reason: _output(result));
        expect(checkout.model('a.bin').readAsBytesSync(), checkout.publicBytes);
        expect(
          checkout.requests.first,
          'https://huggingface.co/owner/pub/resolve/$_revision/files/a.bin '
          'none resume=7',
        );
        expect(part.existsSync(), isFalse, reason: 'moved into place');
      });

      test('a redirect (Hugging Face to its CDN) is followed', () async {
        checkout.serve(
          'owner/pub',
          'files/a.bin.redirect',
          utf8.encode('cdn/xyz'),
        );
        File('${checkout.remote.path}/owner/pub/resolve/$_revision/files/a.bin')
            .deleteSync();
        File('${checkout.remote.path}/cdn/xyz')
          ..createSync(recursive: true)
          ..writeAsBytesSync(checkout.publicBytes);

        final result = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(result.exitCode, 0, reason: _output(result));
        expect(checkout.model('a.bin').readAsBytesSync(), checkout.publicBytes);
        expect(checkout.requests.first, endsWith(' none via=cdn/xyz'));
      });

      test('a refused gated download (401) says to accept the terms', () async {
        File(
          '${checkout.remote.path}/owner/gated/resolve/$_revision/b.bin.status',
        ).writeAsStringSync('401');
        final result = await checkout.run(const [], env: {'HF_TOKEN': _token});
        expect(result.exitCode, 1, reason: _output(result));
        expect(
          result.stderr,
          contains(
            'sub/b.bin: HTTP 401 from owner/gated (gated: accept its terms '
            'with the account of HF_TOKEN)',
          ),
        );
        expect(_output(result), isNot(contains(_token)));
      });
    });
  }
}
