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

import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    var windowFrame = self.frame

    // Measurement runs need a window that is actually on screen and actually
    // the size we claim. macOS stops drawing an occluded window, which shows
    // up as a scene "running at 24 fps" that is really not being drawn at
    // all, and Act 3's whole point is frame timing.
    //   DEMO_WINDOW=1600x1000  set the content size
    //   DEMO_ON_TOP=1          keep the window above everything else
    let env = ProcessInfo.processInfo.environment
    if let size = env["DEMO_WINDOW"] {
      let parts = size.lowercased().split(separator: "x")
      if parts.count == 2, let w = Double(parts[0]), let h = Double(parts[1]) {
        // Content, not frame: a recording at DEMO_WINDOW=1280x720 must come
        // out exactly 1280x720 points, title bar excluded. Pinned, too:
        // something on the dev Mac resizes a fresh window to 929x581 within
        // half a second, which every recording until 2026-10-11 came out at.
        windowFrame = self.frameRect(forContentRect: NSRect(x: windowFrame.origin.x, y: windowFrame.origin.y, width: w, height: h))
        self.styleMask.remove(.resizable)
        self.contentMinSize = NSSize(width: w, height: h)
        self.contentMaxSize = NSSize(width: w, height: h)
      }
    }
    if env["DEMO_ON_TOP"] == "1" {
      self.level = .floating
    }

    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    RegisterGeneratedPlugins(registry: flutterViewController)

    // Come to the front by ourselves instead of leaning on `open` or an
    // AppleScript nudge from the shell, both of which race the first frames.
    //   DEMO_NO_FOCUS=1  show the window on top without taking the keyboard:
    //                    a headless run must not swallow what someone is
    //                    typing elsewhere (Act 3 reads N, Space, A/D and the
    //                    digits as presenter keys).
    if env["DEMO_NO_FOCUS"] == "1" {
      self.orderFrontRegardless()
    } else {
      NSApp.activate(ignoringOtherApps: true)
      self.makeKeyAndOrderFront(nil)
    }

    super.awakeFromNib()
  }
}
