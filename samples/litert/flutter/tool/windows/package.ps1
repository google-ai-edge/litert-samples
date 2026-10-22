# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Packages a Windows release build (the counterpart of tool/linux/package.sh):
#   litert_edge_demos-v<version>-windows-x64\
#     litert_edge_demos.exe, *.dll, data\   the Flutter build (build\windows\x64\runner\Release)
#     run.cmd, run.ps1                     pre-flight launcher (tool\windows)
#     README.md                            how to start it
#     licenses\                            app licence (Apache-2.0), model notices, AGPL-3.0 text + the
#                                          YOLO26n derivation script (the "source" of the bundled AGPL
#                                          model), OpenPhonemizer's Clear BSD licence (the G2P dictionary),
#                                          the notices of the LiteRT-LM native bundle beside the app
#                                          (flutter_edge_ai_litertlm's NOTICES: LiteRT, LiteRT-LM, DirectX
#                                          Shader Compiler, OpenVINO, oneTBB, ...) and flutter_litert's
#                                          licence (libtensorflowlite_c-win.dll, tflite_custom_ops.dll)
#   + a .zip of that folder and its SHA-256 line.
#
# Usage, on the Windows PC that built it (flutter build windows --release), from the repository:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tool\windows\package.ps1 [out_dir]
#
# Windows PowerShell 5.1 compatible, ASCII only (5.1 reads a script without a byte-order mark in the
# ANSI code page).
param([string]$OutDir)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Fail([string]$message) {
  [Console]::Error.WriteLine("error: $message")
  exit 1
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $OutDir) { $OutDir = Join-Path $root 'build\dist' }
$bundle = Join-Path $root 'build\windows\x64\runner\Release'
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'litert_edge_demos.exe'))) {
  Fail "build first: flutter build windows --release ($bundle\litert_edge_demos.exe is missing)"
}

# The built-in models are not in git (tool/fetch_models.sh): package only a build made with the
# verified ones. The same check as `tool/fetch_models.sh --check --dir <bundle models>`, without bash.
$models = Join-Path $bundle 'data\flutter_assets\assets\models'
$total = 0
$bad = @()
foreach ($line in Get-Content -LiteralPath (Join-Path $root 'tool\models.lock')) {
  $fields = @($line.Trim() -split '\s+')
  if ($fields.Count -lt 4 -or ($fields[0] -ne 'hf' -and $fields[0] -ne 'yolo26n-rawhead')) { continue }
  $total++
  $dest = $fields[1]
  $file = Join-Path $models ($dest -replace '/', '\')
  if (-not (Test-Path -LiteralPath $file)) {
    $bad += "MISSING     $dest"
    continue
  }
  $size = (Get-Item -LiteralPath $file).Length
  if ("$size" -ne $fields[2]) {
    $bad += "DIFFERENT   $dest ($size bytes, want $($fields[2]))"
    continue
  }
  $sha = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($sha -ne $fields[3]) { $bad += "DIFFERENT   $dest (SHA-256 differs from the lock)" }
}
if ($total -eq 0) { Fail 'tool\models.lock lists no model' }
if ($bad.Count -gt 0) {
  foreach ($b in $bad) { [Console]::Error.WriteLine("  $b") }
  Fail ("$bundle was built without the verified models ($($bad.Count) of $total): run " +
    'tool/fetch_models.sh (Git Bash), then rebuild')
}
Write-Host "models verified: $total files in $models"

$version = $null
foreach ($line in Get-Content -LiteralPath (Join-Path $root 'pubspec.yaml')) {
  if ($line -match '^version:\s*([0-9.]+)') { $version = $Matches[1]; break }
}
if (-not $version) { Fail 'no version in pubspec.yaml' }
$name = "litert_edge_demos-v$version-windows-x64"
$dest = Join-Path $OutDir $name
if (Test-Path -LiteralPath $dest) { Fail "$dest exists; move it away first" }

# A package's root folder, from the pub resolution of this checkout.
$config = Get-Content -Raw -LiteralPath (Join-Path $root '.dart_tool\package_config.json') | ConvertFrom-Json
function Get-PackageRoot([string]$package) {
  $entry = @($config.packages | Where-Object { $_.name -eq $package })
  if ($entry.Count -eq 0) { Fail "$package is not in .dart_tool\package_config.json (run flutter pub get)" }
  $base = New-Object System.Uri((Join-Path $root '.dart_tool\'))
  return (New-Object System.Uri($base, [string]$entry[0].rootUri)).LocalPath
}
$litertlm = Get-PackageRoot 'flutter_edge_ai_litertlm'
$litertlmNotices = Join-Path $litertlm 'NOTICES'
if (-not (Test-Path -LiteralPath $litertlmNotices)) { Fail "$litertlmNotices is missing" }
$flutterLitertLicense = Join-Path (Get-PackageRoot 'flutter_litert') 'LICENSE'
if (-not (Test-Path -LiteralPath $flutterLitertLicense)) { Fail "$flutterLitertLicense is missing" }

New-Item -ItemType Directory -Path (Join-Path $dest 'licenses') -Force | Out-Null
Copy-Item -Path (Join-Path $bundle '*') -Destination $dest -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run.cmd') -Destination $dest
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run.ps1') -Destination $dest

$utf8 = New-Object System.Text.UTF8Encoding $false
function Write-Text([string]$path, [string[]]$lines) {
  [System.IO.File]::WriteAllText($path, (($lines -join "`r`n") + "`r`n"), $utf8)
}

Write-Text (Join-Path $dest 'README.md') @(
  '# LiteRT Demos for Windows',
  '',
  '    run.cmd               checks the runtime, GPU, NPU, sound, camera and free space, then starts the app',
  '    run.cmd --selftest    headless self-test (hardware, detector, chat model, audio), then a report',
  '',
  '`run.cmd` (it starts `run.ps1`) only warns: the app reports what is missing itself. Everything also goes to',
  '`run.log` next to it. The chat model is a `.litertlm` file you choose in the app (for example',
  '`litert-community/gemma-4-E2B-it-litert-lm` on Hugging Face); the speech, embedding and detector models are',
  'built in. The app needs the latest Microsoft Visual C++ Redistributable (x64). Licences: `licenses\README.md`.'
)

# The app's licence: its own LICENSE, or, inside a repository of samples, the repository's.
$appLicense = $root
while (-not (Test-Path -LiteralPath (Join-Path $appLicense 'LICENSE'))) {
  $parent = Split-Path -Parent $appLicense
  if (-not $parent -or $parent -eq $appLicense) { Fail "no LICENSE in $root or above it" }
  $appLicense = $parent
}
$licenses = Join-Path $dest 'licenses'
Copy-Item -LiteralPath (Join-Path $appLicense 'LICENSE') -Destination (Join-Path $licenses 'LICENSE-app-Apache-2.0.txt')
Copy-Item -LiteralPath (Join-Path $root 'assets\models\NOTICE.md') -Destination (Join-Path $licenses 'MODELS-NOTICE.md')
Copy-Item -LiteralPath (Join-Path $root 'licenses\AGPL-3.0.txt') -Destination (Join-Path $licenses 'AGPL-3.0.txt')
Copy-Item -LiteralPath (Join-Path $root 'licenses\OpenPhonemizer-BSD-3-Clause-Clear.txt') `
  -Destination (Join-Path $licenses 'OpenPhonemizer-BSD-3-Clause-Clear.txt')
Copy-Item -LiteralPath (Join-Path $root 'tool\prune_yolo26n_head.py') -Destination (Join-Path $licenses 'prune_yolo26n_head.py')
Copy-Item -LiteralPath $litertlmNotices -Destination (Join-Path $licenses 'flutter_edge_ai_litertlm-NOTICES.txt')
Copy-Item -LiteralPath $flutterLitertLicense -Destination (Join-Path $licenses 'flutter_litert-LICENSE.txt')
# TODO: the third-party notices of libtensorflowlite_c-win.dll (the TensorFlow Lite C library that
# flutter_litert ships) are not verified; flutter_litert's LICENSE above is the plugin's own.

Write-Text (Join-Path $licenses 'README.md') @(
  '# Licences',
  '',
  '- The app''s own code: Apache-2.0 (`LICENSE-app-Apache-2.0.txt`).',
  '- Bundled models and their terms: `MODELS-NOTICE.md`.',
  '- YOLO26n (object detector) is AGPL-3.0 (`AGPL-3.0.txt`). The bundled file `yolo26n_fp16_rawhead.tflite` is derived',
  '  from Arm''s `yolo26n_conv2d_f16_weights.tflite` (https://huggingface.co/Arm/yolo26n-fp16-litert, itself derived from',
  '  Ultralytics YOLO26) by `prune_yolo26n_head.py` in this folder: it cuts the in-graph selection head and keeps the',
  '  raw [1, 8400, 84] output. Weights are unchanged.',
  '- The speech synthesizer''s G2P dictionary `g2p_dict.txt.gz` (from OpenPhonemizer) is under the Clear BSD License',
  '  (`OpenPhonemizer-BSD-3-Clause-Clear.txt`); the other Matcha G2P files are MIT (`MODELS-NOTICE.md`).',
  '- The LiteRT-LM native runtime beside the app (`LiteRt*.dll`, `libLiteRt*.dll`, `LiteRtLm.dll`, `*webgpu_dawn.dll`,',
  '  `*GemmaModelConstraintProvider.dll`, `StreamProxy.dll`, `LiteRtDispatch.dll`, `dxcompiler.dll`, `dxil.dll`,',
  '  `openvino*.dll`, `tbb*.dll`) comes with flutter_edge_ai_litertlm; its notices are in',
  '  `flutter_edge_ai_litertlm-NOTICES.txt`: LiteRT and LiteRT-LM (Apache-2.0) and their third-party components;',
  '  `dxcompiler.dll` and `dxil.dll` from microsoft/DirectXShaderCompiler under the three licence files of that',
  '  release; the `openvino*.dll`, Intel OpenVINO toolkit redistributables, under the Intel OpenVINO Distribution',
  '  License; the `tbb*.dll` (oneTBB) under Apache-2.0.',
  '- `libtensorflowlite_c-win.dll` and `tflite_custom_ops.dll` come with flutter_litert (Apache-2.0,',
  '  `flutter_litert-LICENSE.txt`).',
  '- The app''s full list of package licences is in the app: home screen, More (the three dots), Licences.'
)

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path -LiteralPath $zip) { Fail "$zip exists; move it away first" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($dest, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
$shaLine = "$hash  $name.zip"
[System.IO.File]::WriteAllText("$zip.sha256", "$shaLine`n", $utf8)
Write-Host $shaLine
Write-Host ('{0}: {1:N0} MB' -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB))
