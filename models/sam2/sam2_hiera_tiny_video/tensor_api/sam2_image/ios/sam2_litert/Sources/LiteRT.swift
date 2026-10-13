import Foundation

final class LiteRTModel {
    private var env: LiteRtEnvironment?
    private var model: LiteRtModel?
    private var options: LiteRtOptions?
    private var compiled: LiteRtCompiledModel?

    enum LiteRTError: Error { case status(String, Int32), notCompiled }

    private func check(_ s: LiteRtStatus, _ what: String) throws {
        if s != kLiteRtStatusOk { throw LiteRTError.status(what, Int32(s.rawValue)) }
    }

    func compileOnGPU(path: String) throws -> (fullyAccelerated: Bool, compileSeconds: Double) {
        try check(LiteRtCreateEnvironment(0, nil, &env), "CreateEnvironment")
        try check(LiteRtCreateModelFromFile(env, path, &model), "CreateModelFromFile")
        try check(LiteRtCreateOptions(&options), "CreateOptions")
        try check(
            LiteRtSetOptionsHardwareAccelerators(
                options, LiteRtHwAcceleratorSet(kLiteRtHwAcceleratorGpu.rawValue)),
            "SetOptionsHardwareAccelerators")

        let start = Date()
        try check(LiteRtCreateCompiledModel(env, model, options, &compiled), "CreateCompiledModel")
        let seconds = Date().timeIntervalSince(start)

        var fully = false
        try check(LiteRtCompiledModelIsFullyAccelerated(compiled, &fully), "IsFullyAccelerated")
        return (fully, seconds)
    }

    func createBuffer(sig: Int, isInput: Bool, index: Int, dims: [Int32]) throws -> LiteRtTensorBuffer? {
        guard let env = env, let compiled = compiled else { throw LiteRTError.notCompiled }
        var reqs: LiteRtTensorBufferRequirements?
        if isInput {
            try check(LiteRtGetCompiledModelInputBufferRequirements(compiled, sig, index, &reqs), "InputReqs")
        } else {
            try check(LiteRtGetCompiledModelOutputBufferRequirements(compiled, sig, index, &reqs), "OutputReqs")
        }
        var bufType: LiteRtTensorBufferType = kLiteRtTensorBufferTypeHostMemory
        try check(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(reqs, 0, &bufType), "BufType")
        var reqBytes: Int = 0
        try check(LiteRtGetTensorBufferRequirementsBufferSize(reqs, &reqBytes), "BufSize")
        var type = dims.withUnsafeBufferPointer {
            Sam2MakeFloat32Type($0.baseAddress, UInt32(dims.count))
        }
        let f32Bytes = dims.reduce(1) { $0 * Int($1) } * MemoryLayout<Float>.stride
        var buf: LiteRtTensorBuffer?
        var status = LiteRtCreateManagedTensorBuffer(env, bufType, &type, reqBytes, &buf)
        if status != kLiteRtStatusOk {
            status = LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &type, max(reqBytes, f32Bytes), &buf)
        }
        try check(status, "CreateManagedBuffer")
        return buf
    }

    func writeFloats(_ buf: LiteRtTensorBuffer?, _ data: [Float]) throws {
        var addr: UnsafeMutableRawPointer?
        try check(LiteRtLockTensorBuffer(buf, &addr, kLiteRtTensorBufferLockModeWrite), "LockWrite")
        _ = data.withUnsafeBytes { memcpy(addr, $0.baseAddress, data.count * MemoryLayout<Float>.stride) }
        try check(LiteRtUnlockTensorBuffer(buf), "UnlockWrite")
    }

    func readFloats(_ buf: LiteRtTensorBuffer?, count: Int) throws -> [Float] {
        var addr: UnsafeMutableRawPointer?
        try check(LiteRtLockTensorBuffer(buf, &addr, kLiteRtTensorBufferLockModeRead), "LockRead")
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { memcpy($0.baseAddress, addr, count * MemoryLayout<Float>.stride) }
        try check(LiteRtUnlockTensorBuffer(buf), "UnlockRead")
        return out
    }

    func runSignature(_ sig: Int, inputs: inout [LiteRtTensorBuffer?], outputs: inout [LiteRtTensorBuffer?]) throws {
        guard let compiled = compiled else { throw LiteRTError.notCompiled }
        try check(LiteRtRunCompiledModel(compiled, sig, inputs.count, &inputs, outputs.count, &outputs), "RunSig")
    }

    deinit {
        if let c = compiled { LiteRtDestroyCompiledModel(c) }
        if let o = options { LiteRtDestroyOptions(o) }
        if let m = model { LiteRtDestroyModel(m) }
        if let e = env { LiteRtDestroyEnvironment(e) }
    }
}
