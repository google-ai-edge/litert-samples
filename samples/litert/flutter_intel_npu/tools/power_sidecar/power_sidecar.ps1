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
# also the app's default on Windows). $Out's folder is reserved for the
# sidecar: it is made admin-only (see Protect-OutputFolder).
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
# This process runs elevated and rewrites $Out four times a second. If an
# ordinary user could write to its folder (any user may create folders under
# ProgramData, and owns what they create), they could swap power.json.tmp for
# a link to a system file, and this script would overwrite that file as
# administrator. So the folder must be the sidecar's own: not a link, nothing
# else in it, owned by Administrators, no inherited rules, users may only read.
function Protect-OutputFolder([string]$dir, [string[]]$ours) {
  # SIDs, not names: group names are translated on non-English Windows.
  $admins = New-Object System.Security.Principal.SecurityIdentifier('S-1-5-32-544')
  $system = New-Object System.Security.Principal.SecurityIdentifier('S-1-5-18')
  $users = New-Object System.Security.Principal.SecurityIdentifier('S-1-5-32-545')
  $me = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  if (Test-Path -LiteralPath $dir) {
    $item = Get-Item -LiteralPath $dir -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
      throw "$dir is a link; give -Out a folder of its own"
    }
    # A folder an ordinary user made is theirs to swap for a link at any
    # moment, even while its rules are being reset: never touch it.
    $owner = (Get-Acl -LiteralPath $dir).GetOwner([System.Security.Principal.SecurityIdentifier])
    if (@($admins.Value, $system.Value, $me.Value) -notcontains $owner.Value) {
      throw "$dir was not created by an administrator; delete it or give -Out another folder"
    }
    $others = @(Get-ChildItem -LiteralPath $dir -Force | Where-Object { $ours -notcontains $_.Name })
    if ($others.Count -gt 0) {
      throw "$dir holds other files; give -Out a folder of its own (its access rules are reset)"
    }
  } else {
    New-Item -ItemType Directory -Path $dir | Out-Null
  }
  $inherit = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
  $none = [System.Security.AccessControl.PropagationFlags]::None
  $allow = [System.Security.AccessControl.AccessControlType]::Allow
  $acl = New-Object System.Security.AccessControl.DirectorySecurity
  $acl.SetOwner($admins)
  $acl.SetAccessRuleProtection($true, $false)
  foreach ($id in $admins, $system) {
    $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($id, 'FullControl', $inherit, $none, $allow)))
  }
  $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($users, 'ReadAndExecute', $inherit, $none, $allow)))
  Set-Acl -LiteralPath $dir -AclObject $acl
  # Whatever was left in it was made before the lock: drop the entries
  # themselves (a link is removed, not followed; Test-Path would follow it
  # and miss a dangling one), then check nothing of ours is left.
  foreach ($name in $ours) {
    $f = Join-Path $dir $name
    try { [IO.File]::Delete($f) } catch {}
    try { if ([IO.Directory]::Exists($f)) { [IO.Directory]::Delete($f) } } catch {}
  }
  $left = @([IO.Directory]::GetFileSystemEntries($dir) | ForEach-Object { Split-Path -Leaf $_ } | Where-Object { $ours -contains $_ })
  if ($left.Count -gt 0) { throw "could not clear $($left -join ', ') in $dir" }
}
Protect-OutputFolder (Split-Path -Parent $Out) @((Split-Path -Leaf $Out), (Split-Path -Leaf "$Out.tmp"))
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
  ($s | ConvertTo-Json -Compress) | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -Force -LiteralPath $tmp -Destination $Out
  Start-Sleep -Milliseconds $IntervalMs
}
