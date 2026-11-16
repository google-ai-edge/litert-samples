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

import CryptoKit
import Foundation

/// The file this app runs: `Resources/models.json`, the same form as the Android app's catalog.
struct ModelEntry {
  var name: String
  var url: URL
  var bytes: UInt64
  var sha256: String
  var card: URL?

  /// `path`: another catalog of the same form (`-modelsJson`), to try a download that goes wrong.
  static func load(path: String? = nil) -> ModelEntry? {
    guard let url = path.map({ URL(fileURLWithPath: $0) }) ?? Bundle.main.url(forResource: "models", withExtension: "json"),
      let data = try? Data(contentsOf: url),
      let catalog = try? JSONDecoder().decode(Catalog.self, from: data),
      let model = catalog.models.first, let file = model.files.first, let fileURL = URL(string: file.url)
    else { return nil }
    return ModelEntry(name: file.name, url: fileURL, bytes: file.bytes, sha256: file.sha256, card: URL(string: model.modelCard))
  }

  private struct Catalog: Decodable {
    struct Model: Decodable {
      var files: [File]
      var modelCard: String
    }
    struct File: Decodable {
      var name: String
      var url: String
      var bytes: UInt64
      var sha256: String
    }
    var models: [Model]
  }
}

/// The model file: under ~/Library/Application Support/ClickOrCode/models, downloaded once with a resume after an
/// interruption, its size and SHA-256 checked before it is renamed into place.
final class ModelStore: NSObject, @unchecked Sendable, URLSessionDataDelegate {
  enum Failure: LocalizedError {
    case http(Int)
    case notHTTPS
    case badResume
    case tooLong
    case noSpace(needed: UInt64, free: UInt64)
    case size(String, UInt64, UInt64)
    case sha(String, String, String)

    var errorDescription: String? {
      switch self {
      case .http(let code): return "The download returned HTTP \(code)."
      case .notHTTPS: return "The download was redirected to a page that is not HTTPS."
      case .badResume: return "The server answered the resume with another part of the file."
      case .tooLong: return "The download is longer than the file should be."
      case .noSpace(let needed, let free):
        return String(format: "The download needs %.1f GB free and %.1f GB is free.", Double(needed) / 1e9, Double(free) / 1e9)
      case .size(let name, let expected, let got):
        return "“\(name)” is \(got) bytes and this app's file is \(expected) bytes. Nothing was loaded."
      case .sha(let name, let expected, let got):
        return "“\(name)” has SHA-256 \(got), not \(expected). Nothing was loaded."
      }
    }
  }

  let entry: ModelEntry
  let directory: URL
  var target: URL { directory.appendingPathComponent(entry.name) }
  var part: URL { directory.appendingPathComponent(entry.name + ".part") }

  private var progress: ((UInt64) -> Void)?
  private var finished: ((Error?) -> Void)?
  private var output: FileHandle?
  private var received: UInt64 = 0
  private var offset: UInt64 = 0
  private var lastReport = Date.distantPast
  private var session: URLSession?
  private var task: URLSessionDataTask?
  private var cancelled = false

  init(entry: ModelEntry, directory: URL? = nil) {
    self.entry = entry
    self.directory =
      directory
      ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        "Library/Application Support/ClickOrCode/models")
  }

  /// A committed file was hashed before it was renamed into place, so a start only checks its size.
  var hasFile: Bool { Self.size(of: target) == entry.bytes }
  var partialBytes: UInt64 { Self.size(of: part) ?? 0 }

  static func size(of url: URL) -> UInt64? {
    let attributes = try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
    return (attributes?[.size] as? NSNumber)?.uint64Value
  }

  /// The bytes still to come, twice over, as the Android app reserves them.
  func checkSpace() throws {
    let missing = entry.bytes - min(partialBytes, entry.bytes)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let values = try directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
    let free = UInt64(max(values.volumeAvailableCapacityForImportantUsage ?? 0, 0))
    if free < missing * 2 { throw Failure.noSpace(needed: missing * 2, free: free) }
  }

  /// Downloads the missing bytes (`progress` gets the bytes held so far, on a background queue). Ends when the file is
  /// whole; the caller checks it with `verify`.
  func download(progress: @escaping (UInt64) -> Void) async throws {
    guard entry.url.scheme == "https" else { throw Failure.notHTTPS }
    try checkSpace()
    if partialBytes > entry.bytes { try FileManager.default.removeItem(at: part) }
    offset = partialBytes
    received = offset
    self.progress = progress
    cancelled = false
    if offset < entry.bytes {
      do {
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
          finished = { error in
            if let error { done.resume(throwing: error) } else { done.resume() }
          }
          var request = URLRequest(url: entry.url, timeoutInterval: 30)
          request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
          if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
          let session = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: nil)
          self.session = session
          let task = session.dataTask(with: request)
          self.task = task
          task.resume()
        }
      } catch Failure.tooLong {
        try? FileManager.default.removeItem(at: part)
        throw Failure.tooLong
      }
    }
    // A whole answer of another size: what arrived is not this file, and a resume could not mend it.
    let got = partialBytes
    guard got == entry.bytes else {
      try? FileManager.default.removeItem(at: part)
      throw Failure.size(entry.name, entry.bytes, got)
    }
  }

  /// Stops a download; what has arrived stays for the next one.
  func cancelDownload() {
    cancelled = true
    task?.cancel()
  }

  func urlSession(
    _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    do {
      guard let http = response as? HTTPURLResponse else { throw Failure.http(0) }
      guard http.url?.scheme == "https" else { throw Failure.notHTTPS }
      guard http.statusCode == 200 || http.statusCode == 206 else { throw Failure.http(http.statusCode) }
      if http.statusCode == 206 {
        let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
        guard range.hasPrefix("bytes \(offset)-"), range.hasSuffix("/\(entry.bytes)") else { throw Failure.badResume }
      } else {
        offset = 0
        received = 0
      }
      if offset == 0 {
        FileManager.default.createFile(atPath: part.path, contents: nil)
      }
      let handle = try FileHandle(forWritingTo: part)
      try handle.truncate(atOffset: offset)
      try handle.seekToEnd()
      output = handle
      completionHandler(.allow)
    } catch {
      completionHandler(.cancel)
      end(error)
    }
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    do {
      guard received + UInt64(data.count) <= entry.bytes else { throw Failure.tooLong }
      try output?.write(contentsOf: data)
      received += UInt64(data.count)
      let now = Date()
      if now.timeIntervalSince(lastReport) >= 0.1 {
        lastReport = now
        progress?(received)
      }
    } catch {
      dataTask.cancel()
      end(error)
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    end(error)
  }

  private func end(_ error: Error?) {
    guard let finished else { return }
    self.finished = nil
    try? output?.close()
    output = nil
    session?.finishTasksAndInvalidate()
    session = nil
    task = nil
    finished(cancelled ? CancellationError() : error)
  }

  /// Hashes the downloaded bytes and renames them into place. A file that does not match is removed.
  func commit() async throws {
    do {
      try await Self.verify(part, entry: entry)
    } catch {
      try? FileManager.default.removeItem(at: part)
      throw error
    }
    try? FileManager.default.removeItem(at: target)
    try FileManager.default.moveItem(at: part, to: target)
  }

  /// A file of the catalog's size and SHA-256 is copied into place (on the same disk, a copy takes no room), so that
  /// the app can run a file already downloaded with `hf download`.
  func importFile(_ url: URL) async throws {
    try await Self.verify(url, entry: entry)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try? FileManager.default.removeItem(at: target)
    try FileManager.default.copyItem(at: url.resolvingSymlinksInPath(), to: target)
  }

  /// Checks a file's size, then its SHA-256.
  static func verify(_ url: URL, entry: ModelEntry) async throws {
    let name = url.lastPathComponent
    let size = Self.size(of: url) ?? 0
    guard size == entry.bytes else { throw Failure.size(name, entry.bytes, size) }
    let sha = try await Task.detached(priority: .userInitiated) { try sha256(of: url) }.value
    guard sha == entry.sha256 else { throw Failure.sha(name, entry.sha256, sha) }
  }

  static func sha256(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
    defer { try? handle.close() }
    var hash = SHA256()
    while true {
      let done = try autoreleasepool { () -> Bool in
        guard let chunk = try handle.read(upToCount: 16 << 20), !chunk.isEmpty else { return true }
        hash.update(data: chunk)
        return false
      }
      if done { break }
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

enum MemoryNeed {
  /// The most memory the engine process used in this app (macOS footprint, Activity Monitor's Memory column), rounded up
  /// to a whole GB.
  static let engineBytes: UInt64 = 7 << 30
  /// Memory left for macOS and the other apps while the engine runs.
  static let reserveBytes: UInt64 = 6 << 30
}
