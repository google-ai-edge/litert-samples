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

# Power / utilisation sidecar for the NPU demo (Windows).
#
# Writes ONE JSON object per second to $Out (atomic rename); the Flutter app
# polls it (POWER_FILE). Two sources, always both:
#   - watts (LibreHardwareMonitor lib): CPU package / cores / platform, GPU.
#     RAPL reads 0 -> null while VBS / Memory Integrity blocks MSR access;
#     they become real numbers once HVCI is off (reboot).
#   - utilisation (Windows "GPU Engine" perf counters, what Task Manager
#     shows): per WDDM adapter, summed per engine type. Adapters are
#     classified by their engine set, because LUIDs change every boot:
#       * has 3d + videodecode (+ compute)  -> the GPU (Intel Arc)
#       * has ONLY a compute engine         -> the NPU (Intel AI Boost is a
#                                              compute-only WDDM adapter)
#       * anything else (a 3d-only stub)    -> ignored
# Run it from an elevated (administrator) PowerShell so LHM can load its
# kernel driver. Keep the script and the DLL in a folder only administrators
# can write to (e.g. under Program Files): an elevated process that loads a
# DLL from a user-writable folder hands that folder's writers admin rights.
# -LibSha256 refuses a DLL that is not the one you downloaded.
#   powershell -ExecutionPolicy Bypass -File power_sidecar.ps1 `
#     -Lib 'C:\Program Files\LibreHardwareMonitor\LibreHardwareMonitorLib.dll'
# Without -Lib (or if it fails to load) only the loads are reported, with
# lhm_error saying why. The app reads $Out (POWER_FILE; the default below is
# also the app's default on Windows).
param(
  [string]$Lib = '',
  [string]$LibSha256 = '',
  [string]$Out = (Join-Path $env:ProgramData 'intel_npu_demo\power.json'),
  [int]$IntervalMs = 250
)
$lhm = $null
$lhmError = $null
if (-not $Lib) {
  $lhmError = 'no -Lib given'
} else {
  try {
    if ($LibSha256 -and ((Get-FileHash -Algorithm SHA256 $Lib).Hash -ne $LibSha256.ToUpper())) {
      throw "SHA-256 of $Lib does not match -LibSha256"
    }
    Add-Type -Path $Lib
    $lhm = New-Object LibreHardwareMonitor.Hardware.Computer
    $lhm.IsCpuEnabled = $true; $lhm.IsGpuEnabled = $true
    $lhm.Open()
  } catch { $lhm = $null; $lhmError = "$_" }
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
$tmp = "$Out.tmp"
function NullIfZero($v) { if ($null -eq $v -or $v -le 0) { $null } else { [math]::Round([double]$v, 2) } }
$rx = 'luid_0x[0-9A-Fa-f]+_(0x[0-9A-Fa-f]+)_phys_\d+_eng_\d+_engtype_(\w+)'
while ($true) {
  $s = [ordered]@{ ts = (Get-Date).ToUniversalTime().ToString('o'); source = 'lhm+pdh' }
  # --- watts (LHM)
  if ($lhm) {
    foreach ($h in $lhm.Hardware) {
      $h.Update()
      foreach ($sen in $h.Sensors) {
        switch ("$($h.HardwareType)|$($sen.SensorType)|$($sen.Name)") {
          'Cpu|Power|CPU Package'   { $s.cpu_package_w  = NullIfZero $sen.Value }
          'Cpu|Power|CPU Cores'     { $s.cpu_cores_w    = NullIfZero $sen.Value }
          'Cpu|Power|CPU Platform'  { $s.cpu_platform_w = NullIfZero $sen.Value }
          'Cpu|Load|CPU Total'      { $s.cpu_load = [math]::Round([double]$sen.Value, 1) }
          'GpuIntel|Power|GPU Power' { $s.gpu_w = NullIfZero $sen.Value }
        }
      }
    }
  }
  # Watts that read 0 with LHM loaded: RAPL is blocked (VBS / Memory
  # Integrity on). Without LHM there is nothing to conclude.
  if ($lhm) { $s.vbs_blocks_rapl = ($null -eq $s.cpu_package_w) } else { $s.lhm_error = $lhmError }
  # --- utilisation (PDH GPU Engine counters)
  $byLuid = @{}
  try {
    $samples = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples
    foreach ($c in $samples) {
      if ($c.InstanceName -match $rx) {
        $luid = $Matches[1].ToLower(); $eng = $Matches[2].ToLower()
        if (-not $byLuid.ContainsKey($luid)) { $byLuid[$luid] = @{} }
        if (-not $byLuid[$luid].ContainsKey($eng)) { $byLuid[$luid][$eng] = 0.0 }
        $byLuid[$luid][$eng] += [double]$c.CookedValue
      }
    }
  } catch { $s.pdh_error = "$_" }
  $gpuLuid = $null; $npuLuid = $null
  foreach ($luid in $byLuid.Keys) {
    $engs = $byLuid[$luid].Keys
    if (($engs -contains '3d') -and ($engs -contains 'videodecode')) { $gpuLuid = $luid }
    elseif (($engs.Count -eq 1) -and ($engs -contains 'compute')) { $npuLuid = $luid }
  }
  if ($gpuLuid) {
    $g = $byLuid[$gpuLuid]
    $s.gpu_luid = $gpuLuid
    $s.gpu_3d_load = [math]::Round([math]::Min(100, $g['3d']), 1)
    $s.gpu_compute_load = [math]::Round([math]::Min(100, $(if ($g.ContainsKey('compute')) { $g['compute'] } else { 0 })), 1)
  }
  if ($npuLuid) {
    $s.npu_luid = $npuLuid
    $s.npu_load = [math]::Round([math]::Min(100, $byLuid[$npuLuid]['compute']), 1)
  }
  ($s | ConvertTo-Json -Compress) | Set-Content -Path $tmp -Encoding UTF8
  Move-Item -Force $tmp $Out
  Start-Sleep -Milliseconds $IntervalMs
}
