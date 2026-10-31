// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


import Darwin
import Foundation

enum SystemMemory {
  private static let assumedKey = "assumePhysicalMemoryGB"

  /// RAM installed in this Mac. Launching with `-assumePhysicalMemoryGB <n>` pretends the Mac has
  /// n GB, which is how the not-enough-memory screen can be tried on a Mac that has enough.
  static var physicalBytes: UInt64 {
    let assumed = UserDefaults.standard.double(forKey: assumedKey)
    if assumed > 0 { return UInt64(assumed * 1_073_741_824) }
    return ProcessInfo.processInfo.physicalMemory
  }

  static var physicalIsAssumed: Bool {
    UserDefaults.standard.double(forKey: assumedKey) > 0
  }

  /// Memory macOS can hand out now without pushing other apps to swap:
  /// free + inactive + speculative + purgeable pages.
  static var availableBytes: UInt64? {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(
      MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
    let result = withUnsafeMutablePointer(to: &stats) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return nil }
    let pages =
      UInt64(stats.free_count) + UInt64(stats.inactive_count)
      + UInt64(stats.speculative_count) + UInt64(stats.purgeable_count)
    return pages * UInt64(vm_kernel_page_size)
  }

  /// A process's physical memory footprint, now and the most it has used: the numbers Activity
  /// Monitor's Memory column and `/usr/bin/time -l` ("peak memory footprint") report.
  static func footprint(pid: pid_t) -> (current: UInt64, peak: UInt64)? {
    var usage = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &usage) {
      $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
        proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
      }
    }
    guard result == 0 else { return nil }
    return (usage.ri_phys_footprint, usage.ri_lifetime_max_phys_footprint)
  }
}

enum Format {
  /// Decimal units, as Finder shows file sizes: 15,379,451,056 bytes → "15.4 GB".
  static func gb(_ bytes: UInt64) -> String {
    if bytes >= 100_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
    return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
  }

  /// Memory the way macOS shows it (Activity Monitor, About This Mac): binary gigabytes written "GB",
  /// 27,917,287,424 bytes -> "26.0 GB".
  static func memory(_ bytes: UInt64) -> String {
    String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
  }

  /// Installed memory the way Apple names it (binary gigabytes, whole): 128 GiB → "128 GB".
  static func ram(_ bytes: UInt64) -> String {
    "\(Int((Double(bytes) / 1_073_741_824).rounded())) GB"
  }

  /// 231.4 → "3:51".
  static func clock(_ seconds: Double) -> String {
    let s = max(0, Int(seconds.rounded(.down)))
    return String(format: "%d:%02d", s / 60, s % 60)
  }
}
