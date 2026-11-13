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

import Foundation
import os

/// The app's memory as iOS counts it against the app's limit.
enum Memory {
  /// Bytes the app can still use before iOS stops it: its memory limit minus its footprint.
  /// Launching with `-assumeAvailableMB <n>` replaces the answer with n MB, so that the
  /// not-enough-memory message can be tried on a phone that has enough (see the README).
  static func available() -> Int64 {
    if let assumed = assumedMB { return assumed * 1_048_576 }
    return Int64(os_proc_available_memory())
  }

  static var availableIsAssumed: Bool { assumedMB != nil }

  private static var assumedMB: Int64? {
    guard let text = LaunchArgument.value("assumeAvailableMB"), let mb = Int64(text), mb > 0 else {
      return nil
    }
    return mb
  }

  /// The app's footprint now, and the most it has been since launch.
  static func footprint() -> (now: Int64, peak: Int64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return (0, 0) }
    return (Int64(info.phys_footprint), info.ledger_phys_footprint_peak)
  }

  /// Memory in binary gigabytes: 1,610,612,736 bytes -> "1.5 GB".
  static func text(_ bytes: Int64) -> String {
    String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
  }
}
