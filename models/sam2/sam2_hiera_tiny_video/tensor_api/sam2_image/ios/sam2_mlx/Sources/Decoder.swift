import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - shared blocks

/// LayerNorm over the channel axis of an NCHW tensor.
final class LayerNorm2d: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray
    let eps: Float
    init(_ channels: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = (x - mean).square().mean(axis: 1, keepDims: true)
        let normed = (x - mean) * rsqrt(variance + eps)
        return weight.reshaped([1, -1, 1, 1]) * normed + bias.reshaped([1, -1, 1, 1])
    }
}

final class SamMLP: Module {
    @ModuleInfo var layers: [Linear]
    let sigmoidOutput: Bool
    let useGelu: Bool
    init(_ inputDim: Int, _ hiddenDim: Int, _ outputDim: Int, _ numLayers: Int,
         sigmoidOutput: Bool = false, gelu: Bool = false) {
        var dims = [inputDim]
        for _ in 0 ..< (numLayers - 1) { dims.append(hiddenDim) }
        dims.append(outputDim)
        self._layers.wrappedValue = (0 ..< numLayers).map { Linear(dims[$0], dims[$0 + 1]) }
        self.sigmoidOutput = sigmoidOutput
        self.useGelu = gelu
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            if i < layers.count - 1 { h = useGelu ? gelu(h) : relu(h) }
        }
        return sigmoidOutput ? sigmoid(h) : h
    }
}

final class SamAttention: Module {
    let numHeads: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    init(embeddingDim: Int, numHeads: Int, downsampleRate: Int = 1) {
        let internalDim = embeddingDim / downsampleRate
        self.numHeads = numHeads
        self._qProj.wrappedValue = Linear(embeddingDim, internalDim)
        self._kProj.wrappedValue = Linear(embeddingDim, internalDim)
        self._vProj.wrappedValue = Linear(embeddingDim, internalDim)
        self._outProj.wrappedValue = Linear(internalDim, embeddingDim)
    }
    private func sep(_ x: MLXArray) -> MLXArray {
        let (b, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        return x.reshaped([b, n, numHeads, c / numHeads]).transposed(0, 2, 1, 3)
    }
    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let qh = sep(qProj(q))
        let kh = sep(kProj(k))
        let vh = sep(vProj(v))
        let scale = 1.0 / sqrt(Float(qh.dim(-1)))
        var out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
        let (b, h, n, c) = (out.dim(0), out.dim(1), out.dim(2), out.dim(3))
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, h * c])
        return outProj(out)
    }
}

final class TwoWayAttentionBlock: Module {
    let skipFirstPE: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: SamAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var crossTI: SamAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: SamMLP
    @ModuleInfo var norm3: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var crossIT: SamAttention
    @ModuleInfo var norm4: LayerNorm
    init(skipFirstPE: Bool) {
        self.skipFirstPE = skipFirstPE
        self._selfAttn.wrappedValue = SamAttention(embeddingDim: 256, numHeads: 8)
        self._norm1.wrappedValue = LayerNorm(dimensions: 256)
        self._crossTI.wrappedValue = SamAttention(embeddingDim: 256, numHeads: 8, downsampleRate: 2)
        self._norm2.wrappedValue = LayerNorm(dimensions: 256)
        self._mlp.wrappedValue = SamMLP(256, 2048, 256, 2)
        self._norm3.wrappedValue = LayerNorm(dimensions: 256)
        self._crossIT.wrappedValue = SamAttention(embeddingDim: 256, numHeads: 8, downsampleRate: 2)
        self._norm4.wrappedValue = LayerNorm(dimensions: 256)
    }
    func callAsFunction(_ queriesIn: MLXArray, _ keysIn: MLXArray, _ queryPE: MLXArray, _ keyPE: MLXArray) -> (MLXArray, MLXArray) {
        var queries = queriesIn
        if skipFirstPE {
            queries = selfAttn(queries, queries, queries)
        } else {
            let q = queries + queryPE
            queries = queries + selfAttn(q, q, queries)
        }
        queries = norm1(queries)
        var q = queries + queryPE
        var k = keysIn + keyPE
        queries = norm2(queries + crossTI(q, k, keysIn))
        queries = norm3(queries + mlp(queries))
        q = queries + queryPE
        k = keysIn + keyPE
        let keys = norm4(keysIn + crossIT(k, q, queries))
        return (queries, keys)
    }
}

final class TwoWayTransformer: Module {
    @ModuleInfo var layers: [TwoWayAttentionBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttn: SamAttention
    @ModuleInfo(key: "norm_final_attn") var normFinal: LayerNorm
    override init() {
        self._layers.wrappedValue = [TwoWayAttentionBlock(skipFirstPE: true), TwoWayAttentionBlock(skipFirstPE: false)]
        self._finalAttn.wrappedValue = SamAttention(embeddingDim: 256, numHeads: 8, downsampleRate: 2)
        self._normFinal.wrappedValue = LayerNorm(dimensions: 256)
    }
    func callAsFunction(_ imageEmbedding: MLXArray, _ imagePE: MLXArray, _ pointEmbedding: MLXArray) -> (MLXArray, MLXArray) {
        let (b, c, h, w) = (imageEmbedding.dim(0), imageEmbedding.dim(1), imageEmbedding.dim(2), imageEmbedding.dim(3))
        var keys = imageEmbedding.reshaped([b, c, h * w]).transposed(0, 2, 1)
        let imagePEflat = imagePE.reshaped([b, c, h * w]).transposed(0, 2, 1)
        var queries = pointEmbedding
        for layer in layers {
            (queries, keys) = layer(queries, keys, pointEmbedding, imagePEflat)
        }
        let q = queries + pointEmbedding
        let k = keys + imagePEflat
        queries = normFinal(queries + finalAttn(q, k, keys))
        return (queries, keys)
    }
}

// MARK: - prompt encoder

final class PositionEmbeddingRandom: Module {
    @ParameterInfo(key: "positional_encoding_gaussian_matrix") var gaussian: MLXArray
    init(numPosFeats: Int = 128) {
        self._gaussian.wrappedValue = MLXArray.zeros([2, numPosFeats])
    }
    func peEncoding(_ coordsIn: MLXArray) -> MLXArray {
        var coords = 2 * coordsIn - 1
        coords = coords.matmul(gaussian)
        coords = 2 * Float.pi * coords
        return concatenated([sin(coords), cos(coords)], axis: -1)
    }
    func forwardWithCoords(_ coordsInput: MLXArray, _ imageSize: (Int, Int)) -> MLXArray {
        let scale = MLXArray([Float(imageSize.1), Float(imageSize.0)])
        let coords = coordsInput.asType(.float32) / scale
        return peEncoding(coords)
    }
    /// Dense positional encoding grid of shape [C, H, W].
    func denseGrid(_ size: (Int, Int)) -> MLXArray {
        let (h, w) = size
        let y = (MLXArray(0 ..< h).asType(.float32) + 0.5) / Float(h)
        let x = (MLXArray(0 ..< w).asType(.float32) + 0.5) / Float(w)
        let yy = broadcast(y.reshaped([h, 1]), to: [h, w])
        let xx = broadcast(x.reshaped([1, w]), to: [h, w])
        let pe = peEncoding(stacked([xx, yy], axis: -1))
        return pe.transposed(2, 0, 1)
    }
}

final class PromptEncoder: Module {
    let embedDim = 256
    let imageEmbeddingSize: (Int, Int)
    let inputImageSize: (Int, Int)
    @ModuleInfo(key: "pe_layer") var peLayer: PositionEmbeddingRandom
    @ModuleInfo(key: "point_embeddings") var pointEmbeddings: [Embedding]
    @ModuleInfo(key: "not_a_point_embed") var notAPoint: Embedding
    @ModuleInfo(key: "no_mask_embed") var noMask: Embedding
    // Mask-downscaling weights exist in the checkpoint but are unused on the point-only path.
    @ModuleInfo(key: "mask_downscaling_0") var maskDown0: Conv2d
    @ModuleInfo(key: "mask_downscaling_1") var maskDown1: LayerNorm2d
    @ModuleInfo(key: "mask_downscaling_3") var maskDown3: Conv2d
    @ModuleInfo(key: "mask_downscaling_4") var maskDown4: LayerNorm2d
    @ModuleInfo(key: "mask_downscaling_6") var maskDown6: Conv2d

    init(imageSize: Int) {
        self.imageEmbeddingSize = (imageSize / 16, imageSize / 16)
        self.inputImageSize = (imageSize, imageSize)
        self._peLayer.wrappedValue = PositionEmbeddingRandom(numPosFeats: 128)
        self._pointEmbeddings.wrappedValue = (0 ..< 4).map { _ in Embedding(embeddingCount: 1, dimensions: 256) }
        self._notAPoint.wrappedValue = Embedding(embeddingCount: 1, dimensions: 256)
        self._noMask.wrappedValue = Embedding(embeddingCount: 1, dimensions: 256)
        self._maskDown0.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 4, kernelSize: 2, stride: 2)
        self._maskDown1.wrappedValue = LayerNorm2d(4)
        self._maskDown3.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 16, kernelSize: 2, stride: 2)
        self._maskDown4.wrappedValue = LayerNorm2d(16)
        self._maskDown6.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 256, kernelSize: 1)
    }

    func denseePE() -> MLXArray {
        peLayer.denseGrid(imageEmbeddingSize).expandedDimensions(axis: 0)
    }

    private func embedPoints(_ pointsIn: MLXArray, _ labelsIn: MLXArray) -> MLXArray {
        var points = pointsIn + 0.5
        let padPoint = MLXArray.zeros([points.dim(0), 1, 2])
        let padLabel = -MLXArray.ones([labelsIn.dim(0), 1]).asType(labelsIn.dtype)
        points = concatenated([points, padPoint], axis: 1)
        let labels = concatenated([labelsIn, padLabel], axis: 1)
        var pe = peLayer.forwardWithCoords(points, inputImageSize)
        let lab = labels.expandedDimensions(axis: -1)
        pe = which(lab .== -1, broadcast(notAPoint.weight, to: pe.shape), pe)
        pe = which(lab .== 0, pe + pointEmbeddings[0].weight, pe)
        pe = which(lab .== 1, pe + pointEmbeddings[1].weight, pe)
        pe = which(lab .== 2, pe + pointEmbeddings[2].weight, pe)
        pe = which(lab .== 3, pe + pointEmbeddings[3].weight, pe)
        return pe
    }

    /// Point-only prompt: returns (sparse, dense).
    func callAsFunction(_ pointCoords: MLXArray, _ pointLabels: MLXArray) -> (MLXArray, MLXArray) {
        let bs = pointCoords.dim(0)
        let sparse = embedPoints(pointCoords, pointLabels)
        let dense = broadcast(noMask.weight.reshaped([1, -1, 1, 1]),
                              to: [bs, embedDim, imageEmbeddingSize.0, imageEmbeddingSize.1])
        return (sparse, dense)
    }
}

// MARK: - mask decoder

final class MaskDecoder: Module {
    @ModuleInfo var transformer: TwoWayTransformer
    @ModuleInfo(key: "iou_token") var iouToken: Embedding
    @ModuleInfo(key: "mask_tokens") var maskTokens: Embedding
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Embedding
    @ModuleInfo(key: "output_upscaling_0") var upscaling0: ConvTransposed2d
    @ModuleInfo(key: "output_upscaling_1") var upscaling1: LayerNorm2d
    @ModuleInfo(key: "output_upscaling_3") var upscaling3: ConvTransposed2d
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d
    @ModuleInfo(key: "output_hypernetworks_mlps") var hyperMLPs: [SamMLP]
    @ModuleInfo(key: "iou_prediction_head") var iouHead: SamMLP
    @ModuleInfo(key: "pred_obj_score_head") var objScoreHead: SamMLP

    override init() {
        self._transformer.wrappedValue = TwoWayTransformer()
        self._iouToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: 256)
        self._maskTokens.wrappedValue = Embedding(embeddingCount: 4, dimensions: 256)
        self._objScoreToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: 256)
        self._upscaling0.wrappedValue = ConvTransposed2d(inputChannels: 256, outputChannels: 64, kernelSize: 2, stride: 2)
        self._upscaling1.wrappedValue = LayerNorm2d(64)
        self._upscaling3.wrappedValue = ConvTransposed2d(inputChannels: 64, outputChannels: 32, kernelSize: 2, stride: 2)
        self._convS0.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 32, kernelSize: 1)
        self._convS1.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 64, kernelSize: 1)
        self._hyperMLPs.wrappedValue = (0 ..< 4).map { _ in SamMLP(256, 256, 32, 3) }
        self._iouHead.wrappedValue = SamMLP(256, 256, 4, 3, sigmoidOutput: true)
        self._objScoreHead.wrappedValue = SamMLP(256, 256, 1, 3)
    }

    func projectHighRes(_ fpn: [MLXArray]) -> [MLXArray] {
        [convNCHW(convS0, fpn[0]), convNCHW(convS1, fpn[1])]
    }

    /// Returns (3 multimask logits [B,3,S/4,S/4], iou_scores [B,3], object_score_logits [B,1]).
    func callAsFunction(_ imageEmbeddings: MLXArray, _ imagePE: MLXArray, _ sparse: MLXArray,
                        _ dense: MLXArray, _ highRes: [MLXArray]) -> (MLXArray, MLXArray, MLXArray) {
        var outputTokens = concatenated([objScoreToken.weight, iouToken.weight, maskTokens.weight], axis: 0)
        outputTokens = broadcast(outputTokens.expandedDimensions(axis: 0),
                                 to: [sparse.dim(0), outputTokens.dim(0), outputTokens.dim(1)])
        let tokens = concatenated([outputTokens, sparse], axis: 1)
        let src = imageEmbeddings + dense
        let posSrc = broadcast(imagePE, to: src.shape)
        let (b, c, h, w) = (src.dim(0), src.dim(1), src.dim(2), src.dim(3))
        let (hs, srcTokens) = transformer(src, posSrc, tokens)
        let iouTokenOut = hs[0..., 1, 0...]
        let objTokenOut = hs[0..., 0, 0...]
        let maskTokensOut = hs[0..., 2 ..< 6, 0...]
        let srcImg = srcTokens.transposed(0, 2, 1).reshaped([b, c, h, w])

        let featS0 = highRes[0]
        let featS1 = highRes[1]
        var x = convTransposeNCHW(upscaling0, srcImg)
        x = gelu(upscaling1(x + featS1))
        x = convTransposeNCHW(upscaling3, x)
        let upscaled = gelu(x + featS0)

        var hyperList: [MLXArray] = []
        for i in 0 ..< 4 { hyperList.append(hyperMLPs[i](maskTokensOut[0..., i, 0...])) }
        let hyper = stacked(hyperList, axis: 1)  // [B, 4, 32]
        let (_, uc, uh, uw) = (upscaled.dim(0), upscaled.dim(1), upscaled.dim(2), upscaled.dim(3))
        let masks = hyper.matmul(upscaled.reshaped([b, uc, uh * uw])).reshaped([b, -1, uh, uw])
        let iou = iouHead(iouTokenOut)[0..., 1 ..< 4]
        let objScore = objScoreHead(objTokenOut)
        return (masks[0..., 1 ..< 4, 0..., 0...], iou, objScore)
    }
}

func convTransposeNCHW(_ conv: ConvTransposed2d, _ x: MLXArray) -> MLXArray {
    conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
}

// MARK: - full image segmenter

final class Sam2ImageSegmenter: Module {
    @ModuleInfo var trunk: Hiera
    @ModuleInfo var neck: FpnNeck
    @ModuleInfo(key: "sam_prompt_encoder") var promptEncoder: PromptEncoder
    @ModuleInfo(key: "sam_mask_decoder") var maskDecoder: MaskDecoder
    @ParameterInfo(key: "no_mem_embed") var noMemEmbed: MLXArray
    let scalp = 1

    let imageSize: Int

    init(config: HieraCfg, imageSize: Int) {
        self.imageSize = imageSize
        self._trunk.wrappedValue = Hiera(
            embedDim: config.embedDim, numHeads: config.numHeads, stages: config.stages,
            globalAttBlocks: config.globalAttBlocks, windowSpec: config.windowSpec,
            qPool: config.qPool, qStride: config.qStride, mlpRatio: config.mlpRatio,
            dimMul: config.dimMul, headMul: config.headMul, posEmbedHW: config.posEmbedHW)
        self._neck.wrappedValue = FpnNeck(backboneChannels: config.backboneChannels, dModel: 256, fpnTopDown: [2, 3])
        self._promptEncoder.wrappedValue = PromptEncoder(imageSize: imageSize)
        self._maskDecoder.wrappedValue = MaskDecoder()
        self._noMemEmbed.wrappedValue = MLXArray.zeros([1, 1, 256])
    }

    /// Returns (vision_features [1,256,64,64], high_res_features).
    func encodeImage(_ pixels: MLXArray) -> (MLXArray, [MLXArray]) {
        var features = neck(trunk(pixels))
        if scalp > 0 { features = Array(features.dropLast(scalp)) }
        let highRes = maskDecoder.projectHighRes(features)
        return (features.last!, highRes)
    }

    /// Returns (3 multimask logits, iou_scores [1,3], object_score_logits [1,1]) for one positive point.
    func predict(_ visionFeatures: MLXArray, _ highRes: [MLXArray], _ coords: MLXArray, _ labels: MLXArray)
        -> (MLXArray, MLXArray, MLXArray) {
        let (sparse, dense) = promptEncoder(coords, labels)
        let imageEmbeddings = visionFeatures + noMemEmbed.transposed(0, 2, 1).reshaped([1, 256, 1, 1])
        return maskDecoder(imageEmbeddings, promptEncoder.denseePE(), sparse, dense, highRes)
    }
}
