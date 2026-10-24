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

# Launcher shipped next to the Windows build (tool/windows/package.ps1 copies it with run.cmd, which
# starts it). Checks what the app needs from the PC, prints what to install or switch on, then starts
# the app with every argument, waits for it and exits with its exit code:
#
#   run.cmd                       the app
#   run.cmd --selftest            the headless self-test, then a report
#
# The checks only warn: the app itself fails with a clear message when something it needs is missing.
#   runtime  the Visual C++ runtime the app links (msvcp140, vcruntime140, vcruntime140_1)
#   gpu      the display adapters (Win32_VideoController); WARP alone (Microsoft Basic Render Driver,
#            Microsoft Basic Display Adapter) means no GPU driver: a GPU mode would run on the CPU
#   npu      an Intel NPU (PnP class ComputeAccelerator) and the Level Zero loader its driver installs
#            (ze_loader.dll); the NPU dispatch beside the app (LiteRtDispatch.dll, openvino*.dll, tbb12.dll)
#   audio    microphones and outputs (PnP class AudioEndpoint) and the microphone privacy switch
#   camera   cameras (PnP classes Camera and Image) and the camera privacy switch
#   temp     free space for %TEMP%: the Intel NPU dispatch writes the weight banks of a model whose NPU
#            blobs carry no weights there (litert_ov_bank_*.bin, several GB), at every load
#
# Everything also goes to run.log next to this script (or %LOCALAPPDATA%\litert_edge_demos\run.log when
# this folder is not writable). Environment:
#   RUN_LOG=PATH        write the log to PATH instead; RUN_LOG=none writes none
#   RUN_PS1_DRY_RUN=1   run the checks, print the app's command line instead of starting it
#
# Windows PowerShell 5.1 compatible (the default on Windows 11), ASCII only: 5.1 reads a script without
# a byte-order mark in the ANSI code page.

$appArgs = @($args)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$app = Join-Path $here 'litert_edge_demos.exe'
$script:warnings = 0
$script:writer = $null
$utf8 = New-Object System.Text.UTF8Encoding $false

function Open-Log([string]$path) {
  try {
    $dir = Split-Path -Parent $path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) {
      New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $w = New-Object System.IO.StreamWriter($path, $true, $utf8)
    $w.AutoFlush = $true
    return $w
  } catch {
    return $null
  }
}

$log = $null
if ($env:RUN_LOG -eq 'none') {
  $log = $null
} elseif ($env:RUN_LOG) {
  $log = $env:RUN_LOG
  $script:writer = Open-Log $log
} else {
  $log = Join-Path $here 'run.log'
  $script:writer = Open-Log $log
  if (-not $script:writer -and $env:LOCALAPPDATA) {
    $log = Join-Path $env:LOCALAPPDATA 'litert_edge_demos\run.log'
    $script:writer = Open-Log $log
  }
}
if (-not $script:writer) { $log = $null }

function Say([string]$line) {
  Write-Host $line
  if ($script:writer) { $script:writer.WriteLine($line) }
}
function Ok([string]$text) { Say "  ok    $text" }
function Warn([string]$text) {
  Say "  WARN  $text"
  $script:warnings++
}
function Hint([string]$text) { Say "        -> $text" }

# Present PnP devices of one class whose status is OK; none when the query fails.
function Get-WorkingDevices([string]$class) {
  @(Get-PnpDevice -Class $class -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { $_.Status -eq 'OK' })
}

# True when Windows denies desktop apps the capability ('microphone', 'webcam'): the device-wide switch
# (HKLM), the user's switch, or the user's switch for desktop (non-packaged) apps.
function Test-ConsentDenied([string]$capability) {
  $base = 'Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\' + $capability
  foreach ($key in @("HKLM:\$base", "HKCU:\$base", "HKCU:\$base\NonPackaged")) {
    $value = (Get-ItemProperty -LiteralPath $key -Name Value -ErrorAction SilentlyContinue).Value
    if ($value -eq 'Deny') { return $true }
  }
  return $false
}

# One check section: a failing query is a warning, never the end of the launcher.
function Invoke-Check([string]$name, [scriptblock]$body) {
  Say $name
  try {
    & $body
  } catch {
    Warn "the $name check failed: $($_.Exception.Message)"
  }
}

Say ('=== litert_edge_demos pre-flight ({0}Z) ===' -f (Get-Date).ToUniversalTime().ToString('s'))

Invoke-Check 'runtime' {
  $system = Join-Path $env:SystemRoot 'System32'
  $missing = @()
  foreach ($dll in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $here $dll)) -and
        -not (Test-Path -LiteralPath (Join-Path $system $dll))) {
      $missing += $dll
    }
  }
  if ($missing.Count -eq 0) {
    Ok 'Visual C++ runtime (msvcp140, vcruntime140, vcruntime140_1)'
  } else {
    Warn "Visual C++ runtime missing: $($missing -join ', '): the app will not start"
    Hint 'install the latest Microsoft Visual C++ Redistributable (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe'
  }
}

Invoke-Check 'gpu' {
  $adapters = @(Get-CimInstance Win32_VideoController -ErrorAction Stop)
  $hardware = 0
  foreach ($adapter in $adapters) {
    $name = "$($adapter.Name)".Trim()
    if ($name -match 'Microsoft Basic (Render Driver|Display Adapter)') {
      Warn "software adapter: $name (WARP: the CPU, not a GPU)"
    } elseif ($name -match 'Microsoft Remote Display Adapter|Microsoft Hyper-V Video') {
      Ok "$name (a virtual display, no compute)"
    } else {
      Ok "GPU: $name, driver $($adapter.DriverVersion)"
      $hardware++
    }
  }
  if ($adapters.Count -eq 0) { Warn 'Windows lists no display adapter (Win32_VideoController)' }
  if ($hardware -eq 0) {
    Hint "no GPU driver: install the GPU vendor's driver (Intel Arc: Intel's graphics driver), or use the CPU"
  }
}

Invoke-Check 'npu' {
  $intel = @(Get-PnpDevice -Class ComputeAccelerator -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { "$($_.Manufacturer) $($_.FriendlyName)" -match 'Intel' })
  if ($intel.Count -eq 0) {
    Warn 'no Intel NPU (PnP class ComputeAccelerator): the NPU mode is not available'
    Hint 'an Intel Core Ultra PC has one: install the Intel NPU driver (Device Manager, Neural processors)'
  }
  foreach ($npu in $intel) {
    if ($npu.Status -eq 'OK') {
      Ok "NPU: $($npu.FriendlyName)"
    } else {
      Warn "NPU $($npu.FriendlyName): status $($npu.Status) (its driver is not working)"
      Hint 'update or reinstall the Intel NPU driver (Device Manager, Neural processors)'
    }
  }
  if (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\ze_loader.dll')) {
    Ok 'ze_loader.dll (Level Zero loader)'
  } else {
    Warn 'ze_loader.dll not found in System32: the NPU dispatch cannot reach the NPU'
    Hint 'install or update the Intel NPU driver; it installs the Level Zero loader'
  }
  $dispatch = @('LiteRtDispatch.dll', 'openvino.dll', 'openvino_intel_npu_plugin.dll', 'tbb12.dll')
  $absent = @($dispatch | Where-Object { -not (Test-Path -LiteralPath (Join-Path $here $_)) })
  $openvino = @(Get-ChildItem -LiteralPath $here -Filter 'openvino*.dll' -ErrorAction SilentlyContinue)
  if ($absent.Count -eq 0) {
    Ok "NPU dispatch beside the app: LiteRtDispatch.dll, $($openvino.Count) openvino*.dll"
  } else {
    Warn "NPU dispatch incomplete beside the app: $($absent -join ', ') missing"
    Hint 'copy the whole build folder (build\windows\x64\runner\Release), not only the .exe'
  }
}

Invoke-Check 'audio' {
  $endpoints = Get-WorkingDevices 'AudioEndpoint'
  # Endpoint ids: SWD\MMDEVAPI\{0.0.1.00000000}.{...} records, {0.0.0.00000000} plays.
  $mics = @($endpoints | Where-Object { $_.InstanceId -match '\{0\.0\.1\.00000000\}' })
  $outputs = @($endpoints | Where-Object { $_.InstanceId -match '\{0\.0\.0\.00000000\}' })
  if ($mics.Count -gt 0) {
    Ok "microphones: $($mics.Count) ($(($mics | ForEach-Object { $_.FriendlyName }) -join '; '))"
  } else {
    Warn 'no working microphone'
    Hint 'plug one in, or enable it: Settings > System > Sound > Input'
  }
  if ($outputs.Count -gt 0) {
    Ok "outputs: $($outputs.Count) ($(($outputs | ForEach-Object { $_.FriendlyName }) -join '; '))"
  } else {
    Warn 'no working audio output'
    Hint 'connect speakers or headphones: Settings > System > Sound > Output'
  }
  if (Test-ConsentDenied 'microphone') {
    Warn 'Windows denies desktop apps the microphone'
    Hint 'Settings > Privacy & security > Microphone: Microphone access and Let desktop apps access your microphone on'
  }
}

Invoke-Check 'camera' {
  $cameras = @(Get-WorkingDevices 'Camera') + @(Get-WorkingDevices 'Image')
  if ($cameras.Count -gt 0) {
    foreach ($camera in $cameras) { Ok "camera: $($camera.FriendlyName)" }
  } else {
    Warn 'no working camera (PnP classes Camera and Image)'
    Hint 'plug in a webcam, or enable it in Device Manager (Cameras)'
  }
  if (Test-ConsentDenied 'webcam') {
    Warn 'Windows denies desktop apps the camera'
    Hint 'Settings > Privacy & security > Camera: Camera access and Let desktop apps access your camera on'
  }
}

Invoke-Check 'temp' {
  $temp = [System.IO.Path]::GetTempPath()
  $drive = New-Object System.IO.DriveInfo([System.IO.Path]::GetPathRoot($temp))
  $free = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
  if ($free -lt 8) {
    Warn "$free GB free for %TEMP% ($temp): an NPU load may need several GB there for weight banks"
    Hint 'free space on that drive, or point TEMP at a drive with more'
  } else {
    Ok "$free GB free for %TEMP% ($temp)"
  }
  $banks = @(Get-ChildItem -LiteralPath $temp -Filter 'litert_ov_bank_*.bin' -ErrorAction SilentlyContinue)
  if ($banks.Count -gt 0) {
    $size = [math]::Round(($banks | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
    Ok "$($banks.Count) NPU weight bank file(s) in %TEMP% ($size GB)"
  }
}

$logNote = ''
if ($log) { $logNote = "; log: $log" }
Say ('=== {0} warning(s){1} ===' -f $script:warnings, $logNote)

if (-not (Test-Path -LiteralPath $app)) {
  Say "ERROR: $app is missing (run.cmd and run.ps1 belong next to the build's litert_edge_demos.exe)"
  if ($script:writer) { $script:writer.Close() }
  exit 127
}

$quoted = @(@($app) + $appArgs | ForEach-Object {
  if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" }
})
if ($env:RUN_PS1_DRY_RUN -eq '1') {
  Say "would run: $($quoted -join ' ')"
  if ($script:writer) { $script:writer.Close() }
  exit 0
}

# The app prints UTF-8; decode its output as such (the console default is the OEM code page).
try { [Console]::OutputEncoding = $utf8 } catch { }
Say "starting: $($quoted -join ' ')"
$ErrorActionPreference = 'Continue'
# Piped, so PowerShell waits for the app (a GUI program) and gets its stdout and stderr. Windows
# PowerShell wraps each stderr line in an ErrorRecord, whose text for an empty line is
# "System.Management.Automation.RemoteException": the line itself is its TargetObject.
# A program Windows refuses to start (Smart App Control, AppLocker, a missing file) throws instead:
# no exit code, so the launcher exits 1 itself.
$code = $null
try {
  & $app @appArgs 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
      if ($_.TargetObject -is [string]) { Say $_.TargetObject } else { Say $_.Exception.Message }
    } else {
      Say "$_"
    }
  }
  $code = $LASTEXITCODE
} catch {
  Say "error: the app did not start: $($_.Exception.Message)"
}
if ($null -eq $code) {
  Say '=== litert_edge_demos did not start ==='
  if ($script:writer) { $script:writer.Close() }
  exit 1
}
# The NTSTATUS codes of a process that never got going or crashed.
$why = switch ($code) {
  -1073741515 { ' (0xC0000135: a DLL it needs is missing; see the runtime check above)' }
  -1073741819 { ' (0xC0000005: it crashed, an access violation; a Visual C++ runtime older than the build does this: install the latest Redistributable)' }
  -1073740791 { ' (0xC0000409: it stopped itself, a fail-fast or a stack buffer overrun)' }
  default { '' }
}
Say "=== litert_edge_demos exited with $code$why ==="
if ($script:writer) { $script:writer.Close() }
exit $code
