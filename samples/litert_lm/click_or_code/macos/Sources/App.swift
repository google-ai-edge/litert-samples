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

import AppKit
import SwiftUI

/// The app's one window (the app holds one engine) and its menus. An AppKit window with SwiftUI inside: on macOS 27 a
/// SwiftUI `Window` scene does not open while the screen is locked, and a recorder run can start then.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
  private let model = AppModel()
  private var window: NSWindow?
  private var actions: MenuActions?

  func applicationDidFinishLaunching(_ notification: Notification) {
    let actions = MenuActions(model: model)
    self.actions = actions
    NSApp.mainMenu = actions.makeMenu()

    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 640, height: 800),
      styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
    window.title = "Click or Code"
    window.isReleasedWhenClosed = false
    let host = NSHostingView(rootView: ContentView(model: model))
    host.sizingOptions = [.minSize]
    window.contentView = host
    window.center()
    Launch.apply(to: window)
    self.window = window
    if Launch.value(after: "--autoplay") == nil {
      window.makeKeyAndOrderFront(nil)
      NSApp.activate(ignoringOtherApps: true)
    } else {
      // A recorder run stays out of the way of whoever is at the Mac: behind the other windows and without the
      // keyboard. Its pictures of the window are of the window alone, so they do not change.
      window.orderBack(nil)
    }
    Task { await Launch.run(model: model) }
  }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    true
  }
}

/// The menu bar.
@MainActor
final class MenuActions: NSObject {
  private let model: AppModel

  init(model: AppModel) {
    self.model = model
  }

  @objc private func openPicture() { model.choosePicture() }
  @objc private func captureScreen() { model.captureScreen() }
  @objc private func captureWindow() { model.captureWindow() }
  @objc private func pastePicture() { model.pastePicture() }

  func makeMenu() -> NSMenu {
    let bar = NSMenu()

    let appMenu = NSMenu()
    appMenu.addItem(withTitle: "Hide Click or Code", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Quit Click or Code", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    add(appMenu, title: "Click or Code", to: bar)

    let file = NSMenu(title: "File")
    item("Open Picture…", #selector(openPicture), "o", [.command], in: file)
    item("Capture Screen", #selector(captureScreen), "1", [.command, .shift], in: file)
    item("Capture Window…", #selector(captureWindow), "2", [.command, .shift], in: file)
    file.addItem(.separator())
    file.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
    add(file, title: "File", to: bar)

    let edit = NSMenu(title: "Edit")
    edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
    edit.addItem(.separator())
    edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
    edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
    edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
    edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
    edit.addItem(.separator())
    item("Paste Picture", #selector(pastePicture), "v", [.command, .shift], in: edit)
    add(edit, title: "Edit", to: bar)

    let windowMenu = NSMenu(title: "Window")
    windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    add(windowMenu, title: "Window", to: bar)
    NSApp.windowsMenu = windowMenu
    return bar
  }

  private func item(
    _ title: String, _ action: Selector, _ key: String, _ modifiers: NSEvent.ModifierFlags, in menu: NSMenu
  ) {
    let entry = menu.addItem(withTitle: title, action: action, keyEquivalent: key)
    entry.keyEquivalentModifierMask = modifiers
    entry.target = self
  }

  private func add(_ menu: NSMenu, title: String, to bar: NSMenu) {
    let entry = bar.addItem(withTitle: title, action: nil, keyEquivalent: "")
    bar.setSubmenu(menu, for: entry)
  }
}

/// Command-line options:
///   --model <path>        run this file, after checking its size and SHA-256, instead of the one the app keeps
///   --picture <path>      show this picture at launch
///   --autoplay <script>   run a recorder script (Autoplay.swift)
///   --run-out <path>      where the recorder writes its run JSON
///   --snapshot-dir <dir>  where the recorder saves pictures of the window
/// and user defaults given as `-key value`:
///   -logDirectory <dir>   where engine logs go (default ~/Library/Logs/ClickOrCode)
///   -runtimeLog info      LiteRT-LM's log level in the engine log (verbose, debug, info, warning, error)
///   -windowFrame "x,y,w,h"  window position and size in points from the top-left of the main screen
///   -windowFloating YES   keep the window above other windows (for screen recordings)
///   -assumePhysicalMemoryGB <n>  pretend the Mac has n GB of memory
///   -modelsJson <path>    another catalog of the form of Resources/models.json, to try a download that goes wrong
///   -modelDirectory <dir> where the model file is kept (default ~/Library/Application Support/ClickOrCode/models)
///   -maxOutputTokens <n>  a smaller cap on a reply, to try a reply that is cut short
///   -noWarmUp YES         do not put a picture through the vision encoder when the engine starts
///   -noEngineCache YES    run the engine without cache files
@MainActor
enum Launch {
  static func run(model: AppModel) async {
    if let picture = value(after: "--picture") { model.setPicture(URL(fileURLWithPath: picture)) }
    if let path = value(after: "--model") { model.modelOverride = URL(fileURLWithPath: path) }
    if let script = value(after: "--autoplay") {
      await Autoplay.run(model: model, scriptPath: script, outPath: value(after: "--run-out"))
      return
    }
    await model.checkModelFile()
  }

  static func value(after flag: String) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
  }

  /// Applies `-windowFrame` and `-windowFloating`.
  static func apply(to window: NSWindow) {
    let defaults = UserDefaults.standard
    if let spec = defaults.string(forKey: "windowFrame"), let screen = NSScreen.screens.first {
      let numbers = spec.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
      if numbers.count == 4 {
        window.setFrame(
          NSRect(x: numbers[0], y: screen.frame.maxY - numbers[1] - numbers[3], width: numbers[2], height: numbers[3]),
          display: true)
      }
    }
    if defaults.bool(forKey: "windowFloating") {
      window.level = .floating
    }
  }

  /// The window's frame in points from the top-left of the main screen (what `screencapture -R` takes).
  static func frameFromTopLeft() -> [Double]? {
    guard let window = NSApp.windows.first(where: { $0.isVisible && $0.title == "Click or Code" }),
      let screen = NSScreen.screens.first
    else { return nil }
    let frame = window.frame
    return [frame.minX, screen.frame.maxY - frame.maxY, frame.width, frame.height]
  }
}
