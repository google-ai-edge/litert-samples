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

// The bicubic resize below follows Pillow 13.0.0 (MIT-CMU License; see ../NOTICE), as the Android app's PilBicubic.kt
// does.
import CryptoKit
import Foundation

/// An 8-bit RGB picture: width x height pixels, row-major, 3 bytes a pixel.
struct Rgb {
  let width: Int
  let height: Int
  let bytes: [UInt8]

  init(width: Int, height: Int, bytes: [UInt8]) {
    precondition(width > 0 && height > 0, "empty picture")
    precondition(bytes.count == width * height * 3, "an RGB \(width)x\(height) picture holds \(width * height * 3) bytes")
    self.width = width
    self.height = height
    self.bytes = bytes
  }

  func resize(_ outWidth: Int, _ outHeight: Int) -> Rgb {
    Rgb(width: outWidth, height: outHeight, bytes: PilBicubic.resize(bytes, width, height, outWidth, outHeight))
  }

  /// Turns the picture upright by its EXIF orientation (1-8), as Pillow's `ImageOps.exif_transpose` does. Any other
  /// value leaves it as it is.
  func oriented(_ orientation: Int) -> Rgb {
    let w = width
    let h = height
    let size: (Int, Int)
    let from: (Int, Int) -> (Int, Int)
    switch orientation {
    case 2: size = (w, h); from = { x, y in (w - 1 - x, y) }  // flip left-right
    case 3: size = (w, h); from = { x, y in (w - 1 - x, h - 1 - y) }  // rotate 180
    case 4: size = (w, h); from = { x, y in (x, h - 1 - y) }  // flip top-bottom
    case 5: size = (h, w); from = { x, y in (y, x) }  // transpose
    case 6: size = (h, w); from = { x, y in (y, h - 1 - x) }  // rotate 90 clockwise
    case 7: size = (h, w); from = { x, y in (w - 1 - y, h - 1 - x) }  // transverse
    case 8: size = (h, w); from = { x, y in (w - 1 - y, x) }  // rotate 90 counter-clockwise
    default: return self
    }
    var out = [UInt8](repeating: 0, count: size.0 * size.1 * 3)
    for y in 0..<size.1 {
      for x in 0..<size.0 {
        let (sx, sy) = from(x, y)
        let s = (sy * w + sx) * 3
        let o = (y * size.0 + x) * 3
        out[o] = bytes[s]
        out[o + 1] = bytes[s + 1]
        out[o + 2] = bytes[s + 2]
      }
    }
    return Rgb(width: size.0, height: size.1, bytes: out)
  }

  var sha256: String {
    SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
  }
}

/// The picture the model file reads: one 1920 x 1088 frame (2,040 image tokens), the grid the model's own processor
/// gives a 1920 x 1080 desktop screenshot. LiteRT-LM resizes every picture to that size without keeping its aspect
/// ratio, so the app makes the frame itself, with Pillow's bicubic filter, as the model card's script does. The model's
/// point is a fraction of the whole frame, so the stretch does not move it relative to what is on the screen.
enum Pictures {
  static let width = 1920
  static let height = 1088

  static func stretch(_ image: Rgb) -> Rgb {
    image.width == width && image.height == height ? image : image.resize(width, height)
  }
}

/// Pillow 13.0.0's `Image.resize(size, Image.BICUBIC)` for 8-bit RGB, pixel for pixel. The model card's script (and the
/// checks behind the card's numbers) prepare every picture with Pillow, so the app prepares the same pixels.
///
/// The algorithm of Pillow's src/libImaging/Resample.c (13.0.0): a bicubic kernel (a = -0.5) widened by the scale when
/// shrinking, coefficients normalized to sum 1 and rounded to 22-bit fixed point, and two passes, each rounded and
/// clipped to 8 bits. The horizontal pass runs first (over only the rows the vertical pass reads), unless the height
/// shrinks by more than twice as many pixels as the width does; then the vertical pass runs first, over every column.
///
/// Pillow: Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and contributors,
/// Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors; MIT-CMU License (../android/licenses/Pillow-MIT-CMU.txt).
enum PilBicubic {
  private static let precisionBits: Int32 = 32 - 8 - 2

  private struct Coeffs {
    let ksize: Int
    var bounds: [Int]
    let kk: [Int32]
  }

  private static func bicubic(_ x0: Double) -> Double {
    let a = -0.5
    let x = x0 < 0 ? -x0 : x0
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0
  }

  /// precompute_coeffs + normalize_coeffs_8bpc for box [0, inSize) -> outSize.
  private static func coeffs(_ inSize: Int, _ outSize: Int) -> Coeffs {
    let scale = Double(Float(inSize)) / Double(outSize)
    let filterscale = scale < 1 ? 1 : scale
    let support = 2 * filterscale
    let ksize = Int(support.rounded(.up)) * 2 + 1
    var pre = [Double](repeating: 0, count: outSize * ksize)
    var bounds = [Int](repeating: 0, count: outSize * 2)
    let ss = 1 / filterscale
    for xx in 0..<outSize {
      let center = (Double(xx) + 0.5) * scale
      var ww = 0.0
      var xmin = Int(center - support + 0.5)
      if xmin < 0 { xmin = 0 }
      var xmax = Int(center + support + 0.5)
      if xmax > inSize { xmax = inSize }
      xmax -= xmin
      let base = xx * ksize
      for x in 0..<xmax {
        let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
        pre[base + x] = w
        ww += w
      }
      if ww != 0 {
        for x in 0..<xmax { pre[base + x] /= ww }
      }
      bounds[xx * 2] = xmin
      bounds[xx * 2 + 1] = xmax
    }
    let one = Double(Int32(1) << precisionBits)
    let kk = pre.map { v in v < 0 ? Int32(-0.5 + v * one) : Int32(0.5 + v * one) }
    return Coeffs(ksize: ksize, bounds: bounds, kk: kk)
  }

  @inline(__always) private static func clip8(_ acc: Int32) -> UInt8 {
    let v = acc >> precisionBits
    return v < 0 ? 0 : v > 255 ? 255 : UInt8(v)
  }

  /// Horizontal pass: `rows` rows of `src` (`srcWidth` wide), starting at row `firstRow`, to `outWidth` columns.
  private static func horizontal(_ src: [UInt8], _ srcWidth: Int, _ firstRow: Int, _ rows: Int, _ outWidth: Int, _ c: Coeffs) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: outWidth * rows * 3)
    let half = Int32(1) << (precisionBits - 1)
    src.withUnsafeBufferPointer { s in
      c.kk.withUnsafeBufferPointer { kk in
        out.withUnsafeMutableBufferPointer { o in
          for yy in 0..<rows {
            let inRow = (yy + firstRow) * srcWidth * 3
            let outRow = yy * outWidth * 3
            for xx in 0..<outWidth {
              let xmin = c.bounds[xx * 2]
              let xmax = c.bounds[xx * 2 + 1]
              let k = xx * c.ksize
              var s0 = half
              var s1 = half
              var s2 = half
              var p = inRow + xmin * 3
              for x in 0..<xmax {
                let w = kk[k + x]
                s0 &+= Int32(s[p]) &* w
                s1 &+= Int32(s[p + 1]) &* w
                s2 &+= Int32(s[p + 2]) &* w
                p += 3
              }
              let i = outRow + xx * 3
              o[i] = clip8(s0)
              o[i + 1] = clip8(s1)
              o[i + 2] = clip8(s2)
            }
          }
        }
      }
    }
    return out
  }

  /// Vertical pass: every column of `src` (`width` wide) to `outHeight` rows.
  private static func vertical(_ src: [UInt8], _ width: Int, _ outHeight: Int, _ c: Coeffs) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: width * outHeight * 3)
    let half = Int32(1) << (precisionBits - 1)
    let stride = width * 3
    src.withUnsafeBufferPointer { s in
      c.kk.withUnsafeBufferPointer { kk in
        out.withUnsafeMutableBufferPointer { o in
          for yy in 0..<outHeight {
            let ymin = c.bounds[yy * 2]
            let ymax = c.bounds[yy * 2 + 1]
            let k = yy * c.ksize
            let outRow = yy * stride
            for xx in 0..<width {
              var s0 = half
              var s1 = half
              var s2 = half
              var p = ymin * stride + xx * 3
              for y in 0..<ymax {
                let w = kk[k + y]
                s0 &+= Int32(s[p]) &* w
                s1 &+= Int32(s[p + 1]) &* w
                s2 &+= Int32(s[p + 2]) &* w
                p += stride
              }
              let i = outRow + xx * 3
              o[i] = clip8(s0)
              o[i + 1] = clip8(s1)
              o[i + 2] = clip8(s2)
            }
          }
        }
      }
    }
    return out
  }

  /// `rgb` = width x height x 3 bytes, row-major; returns outWidth x outHeight x 3 bytes. A dimension that does not
  /// change skips its pass, as in Pillow.
  static func resize(_ rgb: [UInt8], _ width: Int, _ height: Int, _ outWidth: Int, _ outHeight: Int) -> [UInt8] {
    let needHorizontal = outWidth != width
    let needVertical = outHeight != height
    if !needHorizontal && !needVertical { return rgb }
    let horizontalFirst = !((height - outHeight) > 0 && (height - outHeight) > (width - outWidth) * 2)
    var vert = coeffs(height, outHeight)
    if horizontalFirst {
      var src = rgb
      var srcWidth = width
      if needHorizontal {
        let horiz = coeffs(width, outWidth)
        let yFirst = vert.bounds[0]
        let yLast = vert.bounds[outHeight * 2 - 2] + vert.bounds[outHeight * 2 - 1]
        // Shift the vertical bounds to the rows the horizontal pass produces.
        for i in 0..<outHeight { vert.bounds[i * 2] -= yFirst }
        src = horizontal(rgb, width, yFirst, yLast - yFirst, outWidth, horiz)
        srcWidth = outWidth
      }
      return needVertical ? vertical(src, srcWidth, outHeight, vert) : src
    }
    // The height shrinks the most: the vertical pass first, over every column.
    let tall = vertical(rgb, width, outHeight, vert)
    return needHorizontal ? horizontal(tall, width, 0, outHeight, outWidth, coeffs(width, outWidth)) : tall
  }
}
