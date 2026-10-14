import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Hiera trunk

/// Conv over an NCHW input by transposing to NHWC (MLX conv layout) and back.
func convNCHW(_ conv: Conv2d, _ x: MLXArray) -> MLXArray {
    conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
}

func windowPartition(_ x: MLXArray, _ windowSize: Int) -> (MLXArray, (Int, Int)) {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let padH = (windowSize - h % windowSize) % windowSize
    let padW = (windowSize - w % windowSize) % windowSize
    var xp = x
    if padH > 0 || padW > 0 {
        xp = padded(x, widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
    }
    let hp = h + padH
    let wp = w + padW
    xp = xp.reshaped([b, hp / windowSize, windowSize, wp / windowSize, windowSize, c])
    let windows = xp.transposed(0, 1, 3, 2, 4, 5).reshaped([-1, windowSize, windowSize, c])
    return (windows, (hp, wp))
}

func windowUnpartition(_ windows: MLXArray, _ windowSize: Int, _ padHW: (Int, Int), _ hw: (Int, Int)) -> MLXArray {
    let (hp, wp) = padHW
    let (h, w) = hw
    let b = windows.dim(0) / (hp * wp / windowSize / windowSize)
    var x = windows.reshaped([b, hp / windowSize, wp / windowSize, windowSize, windowSize, -1])
    x = x.transposed(0, 1, 3, 2, 4, 5).reshaped([b, hp, wp, -1])
    if hp > h || wp > w {
        x = x[0..., 0 ..< h, 0 ..< w, 0...]
    }
    return x
}

final class HieraMLP: Module {
    @ModuleInfo var layers: [Linear]
    init(_ dim: Int, _ hidden: Int, _ out: Int) {
        self._layers.wrappedValue = [Linear(dim, hidden), Linear(hidden, out)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers[1](gelu(layers[0](x))) }
}

final class PatchEmbed: Module {
    @ModuleInfo var proj: Conv2d
    init(embedDim: Int) {
        self._proj.wrappedValue = Conv2d(inputChannels: 3, outputChannels: embedDim, kernelSize: 7, stride: 4, padding: 3)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x.transposed(0, 2, 3, 1))
    }
}

final class MultiScaleAttention: Module {
    let numHeads: Int
    let qStride: Int?
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    var pool: MaxPool2d?

    init(dim: Int, dimOut: Int, numHeads: Int, qStride: Int?) {
        self.numHeads = numHeads
        self.qStride = qStride
        self._qkv.wrappedValue = Linear(dim, dimOut * 3)
        self._proj.wrappedValue = Linear(dimOut, dimOut)
        if let s = qStride { self.pool = MaxPool2d(kernelSize: [s, s], stride: [s, s]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        var h = x.dim(1)
        var w = x.dim(2)
        let qkvOut = qkv(x).reshaped([b, h * w, 3, numHeads, -1])
        var q = qkvOut[0..., 0..., 0]
        let k = qkvOut[0..., 0..., 1]
        let v = qkvOut[0..., 0..., 2]
        if let pool {
            var qr = q.reshaped([b, h, w, -1])
            qr = pool(qr)
            h = qr.dim(1); w = qr.dim(2)
            q = qr.reshaped([b, h * w, numHeads, -1])
        }
        let scale = 1.0 / sqrt(Float(q.dim(-1)))
        var out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3), scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([b, h, w, -1])
        return proj(out)
    }
}

final class MultiScaleBlock: Module {
    let dim: Int
    let dimOut: Int
    let qStride: Int?
    let windowSize: Int
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: MultiScaleAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: HieraMLP
    @ModuleInfo var proj: Linear?
    var pool: MaxPool2d?

    init(dim: Int, dimOut: Int, numHeads: Int, mlpRatio: Float, qStride: Int?, windowSize: Int) {
        self.dim = dim
        self.dimOut = dimOut
        self.qStride = qStride
        self.windowSize = windowSize
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = MultiScaleAttention(dim: dim, dimOut: dimOut, numHeads: numHeads, qStride: qStride)
        self._norm2.wrappedValue = LayerNorm(dimensions: dimOut, eps: 1e-6)
        self._mlp.wrappedValue = HieraMLP(dimOut, Int(Float(dimOut) * mlpRatio), dimOut)
        if dim != dimOut {
            self._proj.wrappedValue = Linear(dim, dimOut)
            if let s = qStride { self.pool = MaxPool2d(kernelSize: [s, s], stride: [s, s]) }
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var shortcut = x
        var xn = norm1(x)
        if dim != dimOut, let proj, let pool {
            // proj on NHWC tokens, pool on NHWC spatial
            shortcut = pool(proj(xn))
        }
        var windowSizeLocal = windowSize
        var h = 0, w = 0
        var padHW = (0, 0)
        if windowSize > 0 {
            h = xn.dim(1); w = xn.dim(2)
            let (win, pad) = windowPartition(xn, windowSize)
            xn = win; padHW = pad
        }
        xn = attn(xn)
        if let s = qStride {
            windowSizeLocal = windowSize / s
            // Python reassigns h, w to the pooled shortcut dims here.
            h = shortcut.dim(1); w = shortcut.dim(2)
            let padH = (windowSizeLocal - h % windowSizeLocal) % windowSizeLocal
            let padW = (windowSizeLocal - w % windowSizeLocal) % windowSizeLocal
            padHW = (h + padH, w + padW)
        }
        if windowSize > 0 {
            xn = windowUnpartition(xn, windowSizeLocal, padHW, (h, w))
        }
        let merged = shortcut + xn
        return merged + mlp(norm2(merged))
    }
}

final class Hiera: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ParameterInfo(key: "pos_embed_full") var posEmbedFull: MLXArray
    @ModuleInfo var blocks: [MultiScaleBlock]
    let stageEnds: [Int]

    init(embedDim: Int, numHeads: Int, stages: [Int], globalAttBlocks: Set<Int>,
         windowSpec: [Int], qPool: Int, qStride: Int, mlpRatio: Float, dimMul: Float, headMul: Float,
         posEmbedHW: (Int, Int)) {
        self._patchEmbed.wrappedValue = PatchEmbed(embedDim: embedDim)
        self._posEmbedFull.wrappedValue = MLXArray.zeros([1, posEmbedHW.0, posEmbedHW.1, embedDim])
        let depth = stages.reduce(0, +)
        var ends: [Int] = []
        for i in 1...stages.count { ends.append(stages.prefix(i).reduce(0, +) - 1) }
        self.stageEnds = ends
        let qPoolBlocks = Set(ends.dropLast().map { $0 + 1 }.prefix(qPool))

        var blocksArr: [MultiScaleBlock] = []
        var ed = embedDim
        var nh = numHeads
        var curStage = 1
        for i in 0 ..< depth {
            var dimOut = ed
            // window_size uses the CURRENT stage spec, fixed before the stage increment
            // (Python sets it, then increments cur_stage for the next block).
            var ws = windowSpec[curStage - 1]
            if globalAttBlocks.contains(i) { ws = 0 }
            if ends.contains(i - 1) {
                dimOut = Int(Float(ed) * dimMul)
                nh = Int(Float(nh) * headMul)
                curStage += 1
            }
            blocksArr.append(MultiScaleBlock(
                dim: ed, dimOut: dimOut, numHeads: nh, mlpRatio: mlpRatio,
                qStride: qPoolBlocks.contains(i) ? qStride : nil, windowSize: ws))
            ed = dimOut
        }
        self._blocks.wrappedValue = blocksArr
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = patchEmbed(x)
        h = h + posEmbedFull[0..., 0 ..< h.dim(1), 0 ..< h.dim(2), 0...]
        var outputs: [MLXArray] = []
        for (i, block) in blocks.enumerated() {
            h = block(h)
            if stageEnds.contains(i) { outputs.append(h.transposed(0, 3, 1, 2)) }
        }
        return outputs
    }
}

// MARK: - FPN neck

func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    // x is NCHW; repeat along H and W.
    repeated(repeated(x, count: 2, axis: 2), count: 2, axis: 3)
}

final class FpnNeck: Module {
    @ModuleInfo var convs: [Conv2d]
    let fpnTopDown: Set<Int>
    let dModel: Int

    init(backboneChannels: [Int], dModel: Int, fpnTopDown: Set<Int>) {
        self.dModel = dModel
        self.fpnTopDown = fpnTopDown
        self._convs.wrappedValue = backboneChannels.map { Conv2d(inputChannels: $0, outputChannels: dModel, kernelSize: 1) }
    }

    /// Returns backbone_fpn features (NCHW). Position encoding is not needed for the image path.
    func callAsFunction(_ xs: [MLXArray]) -> [MLXArray] {
        let n = convs.count - 1
        var out = [MLXArray?](repeating: nil, count: convs.count)
        var prev: MLXArray? = nil
        for i in stride(from: n, through: 0, by: -1) {
            let lateral = convNCHW(convs[n - i], xs[i])
            if fpnTopDown.contains(i), let p = prev {
                prev = lateral + upsampleNearest2x(p)
            } else {
                prev = lateral
            }
            out[i] = prev
        }
        return out.map { $0! }
    }
}

final class Sam2ImageEncoder: Module {
    @ModuleInfo var trunk: Hiera
    @ModuleInfo var neck: FpnNeck
    let scalp: Int

    init(config: HieraCfg) {
        self._trunk.wrappedValue = Hiera(
            embedDim: config.embedDim, numHeads: config.numHeads, stages: config.stages,
            globalAttBlocks: config.globalAttBlocks, windowSpec: config.windowSpec,
            qPool: config.qPool, qStride: config.qStride, mlpRatio: config.mlpRatio,
            dimMul: config.dimMul, headMul: config.headMul, posEmbedHW: config.posEmbedHW)
        self._neck.wrappedValue = FpnNeck(
            backboneChannels: config.backboneChannels, dModel: 256, fpnTopDown: [2, 3])
        self.scalp = 1
    }

    /// Returns backbone_fpn (NCHW), high-to-low res; last is vision_features 256x64x64.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var features = neck(trunk(x))
        if scalp > 0 { features = Array(features.dropLast(scalp)) }
        return features
    }
}

struct HieraCfg {
    var embedDim = 96
    var numHeads = 1
    var stages = [1, 2, 7, 2]
    var globalAttBlocks: Set<Int> = [5, 7, 9]
    var windowSpec = [8, 4, 14, 7]
    var qPool = 3
    var qStride = 2
    var mlpRatio: Float = 4.0
    var dimMul: Float = 2.0
    var headMul: Float = 2.0
    var posEmbedHW = (256, 256)
    var backboneChannels = [768, 384, 192, 96]
}
