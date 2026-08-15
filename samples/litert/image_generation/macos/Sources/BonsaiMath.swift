// Copyright 2026 Daisuke Majima. All Rights Reserved.
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
// =============================================================================

// Host-side math for the Bonsai pipeline, kept UIKit-free so it can be
// cross-checked on the Mac against generate.py / the device fixtures.

import Foundation

enum BonsaiMath {
    static let seq = 256
    static let tokens = 1024      // 512x512 -> 32x32 patch grid
    static let latGrid = 32
    static let packedChannels = 128

    /// FLUX.2-klein sigma schedule (generate.py flowmatch_sigmas): linspace
    /// shifted by the empirical mu, exponential time-shift, timestep == sigma.
    /// steps=4 reproduces the device-fixture manifest sigmas exactly.
    static func sigmas(steps: Int) -> [Float] {
        let m200 = 0.00016927 * Double(tokens) + 0.45666666
        let m10 = 8.73809524e-05 * Double(tokens) + 1.89833333
        let a = (m200 - m10) / 190.0
        let mu = a * Double(steps) + (m200 - 200.0 * a)
        var out: [Float] = []
        for i in 0..<steps {
            let lin = 1.0 - Double(i) * (1.0 - 1.0 / Double(steps)) / Double(max(steps - 1, 1))
            out.append(Float(exp(mu) / (exp(mu) + (1.0 / lin - 1.0))))
        }
        out.append(0)
        return out
    }

    /// Image-token position ids: [0, h, w, 0] over the 32x32 grid -> (1024, 4).
    static func imgIDs() -> [Float] {
        var out = [Float](repeating: 0, count: tokens * 4)
        for h in 0..<latGrid {
            for w in 0..<latGrid {
                let base = (h * latGrid + w) * 4
                out[base + 1] = Float(h)
                out[base + 2] = Float(w)
            }
        }
        return out
    }

    /// Text-token position ids: [0, 0, 0, i] -> (256, 4).
    static func txtIDs() -> [Float] {
        var out = [Float](repeating: 0, count: seq * 4)
        for i in 0..<seq { out[i * 4 + 3] = Float(i) }
        return out
    }

    /// Seeded standard-normal noise (1, 1024, 128). SplitMix64 + Box-Muller —
    /// a valid N(0,1) draw, intentionally NOT numpy's stream (the seed only
    /// buys in-app reproducibility).
    static func noise(seed: UInt64) -> [Float] {
        var state = seed
        func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        func uniform() -> Double {           // (0, 1], never 0 for log()
            (Double(next() >> 11) + 1.0) / 9007199254740993.0
        }
        let n = tokens * packedChannels
        var out = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            let r = (-2.0 * log(uniform())).squareRoot()
            let theta = 2.0 * Double.pi * uniform()
            out[i] = Float(r * cos(theta))
            if i + 1 < n { out[i + 1] = Float(r * sin(theta)) }
            i += 2
        }
        return out
    }

    /// (1024, 128) packed tokens -> (1, 32, 64, 64) VAE latent: per-PACKED-
    /// channel BN affine first, then the 2x2 patch unfold (packed channel
    /// m = c*4 + i*2 + j lands at z[c, 2h+i, 2w+j] for token (h, w)).
    static func unpatchify(_ lat: [Float], scale: [Float], shift: [Float]) -> [Float] {
        var z = [Float](repeating: 0, count: 32 * 64 * 64)
        for h in 0..<latGrid {
            for w in 0..<latGrid {
                let base = (h * latGrid + w) * packedChannels
                for c in 0..<32 {
                    for i in 0...1 {
                        for j in 0...1 {
                            let m = c * 4 + i * 2 + j
                            z[c * 4096 + (2 * h + i) * 64 + (2 * w + j)] =
                                scale[m] * lat[base + m] + shift[m]
                        }
                    }
                }
            }
        }
        return z
    }
}
