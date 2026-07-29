// Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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

import CoreMedia
import CoreVideo
import Metal
import UIKit
import MetalPerformanceShaders
import MetalKit

class SegmentedImageRenderer {

  var description: String = "Metal"

  var isPrepared = false

  private(set) var outputFormatDescription: CMFormatDescription?

  private var outputPixelBufferPool: CVPixelBufferPool?

  private let metalDevice = MTLCreateSystemDefaultDevice()!

  private var computePipelineState: MTLComputePipelineState?

  private var textureCache: CVMetalTextureCache!
  let context: CIContext

  let textureLoader: MTKTextureLoader

  private lazy var commandQueue: MTLCommandQueue? = {
    return self.metalDevice.makeCommandQueue()
  }()

  required init() {
    let defaultLibrary = metalDevice.makeDefaultLibrary()!
    let kernelFunction = defaultLibrary.makeFunction(name: "mergeColor")
    do {
      computePipelineState = try metalDevice.makeComputePipelineState(function: kernelFunction!)
    } catch {
      print("Could not create pipeline state: \(error)")
    }
    context =  CIContext(mtlDevice: metalDevice)
    textureLoader = MTKTextureLoader(device: metalDevice)
  }

  func prepare(with imageSize: CGSize,
               outputRetainedBufferCountHint: Int) {
    reset()

    (outputPixelBufferPool, _, outputFormatDescription) = allocateOutputBufferPool(with: imageSize,
                                                                                   outputRetainedBufferCountHint: outputRetainedBufferCountHint)
    if outputPixelBufferPool == nil {
      return
    }

    var metalTextureCache: CVMetalTextureCache?
    if CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, metalDevice, nil, &metalTextureCache) != kCVReturnSuccess {
      assertionFailure("Unable to allocate texture cache")
    } else {
      textureCache = metalTextureCache
    }

    isPrepared = true
  }

  func reset() {
    outputPixelBufferPool = nil
    outputFormatDescription = nil
    textureCache = nil
    isPrepared = false
  }

  func render(image: UIImage, segmenterData: Data, maskSize: CGSize) -> UIImage? {

    var newPixelBuffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outputPixelBufferPool!, &newPixelBuffer)
    guard let outputPixelBuffer = newPixelBuffer else {
      print("Allocation failure: Could not get pixel buffer from pool. (\(self.description))")
      return nil
    }
    guard let outputTexture = makeTextureFromCVPixelBuffer(pixelBuffer: outputPixelBuffer, textureFormat: .bgra8Unorm) else {
      return nil
    }

    var inputTexture: MTLTexture!
    do {
      guard let cgImage = image.fixedOrientation() else { return nil }
      inputTexture = try textureLoader.newTexture(cgImage: cgImage)
    } catch {
      print("create texture error: ", error.localizedDescription)
      return nil
    }


    // Set up command queue, buffer, and encoder.
    guard let commandQueue = commandQueue,
          let commandBuffer = commandQueue.makeCommandBuffer(),
          let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      print("Failed to create a Metal command queue.")
      CVMetalTextureCacheFlush(textureCache!, 0)
      return nil
    }

    commandEncoder.label = "Demo Metal"
    commandEncoder.setComputePipelineState(computePipelineState!)
    commandEncoder.setTexture(inputTexture, index: 0)
    commandEncoder.setTexture(outputTexture, index: 1)
    let buffer = segmenterData.withUnsafeBytes { (rawBufferPointer) -> MTLBuffer? in
        guard let baseAddress = rawBufferPointer.baseAddress else { return nil }
        return metalDevice.makeBuffer(
            bytes: baseAddress,
            length: segmenterData.count,
            options: .storageModeShared
        )
    }

    // 4. Ensure the buffer was created
    guard let metalBuffer = buffer else {
        fatalError("Failed to create Metal buffer")
    }
    commandEncoder.setBuffer(metalBuffer, offset: 0, index: 0)
    var markWidth: Int = Int(maskSize.width)
    var markHeight: Int = Int(maskSize.height)
    commandEncoder.setBytes(&markWidth, length: MemoryLayout<Int>.size, index: 1)
    commandEncoder.setBytes(&markHeight, length: MemoryLayout<Int>.size, index: 2)

    // Set up the thread groups.
    let width = computePipelineState!.threadExecutionWidth
    let height = computePipelineState!.maxTotalThreadsPerThreadgroup / width
    let threadsPerThreadgroup = MTLSizeMake(width, height, 1)
    let threadgroupsPerGrid = MTLSize(width: (inputTexture.width + width - 1) / width,
                                      height: (inputTexture.height + height - 1) / height,
                                      depth: 1)
    commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    commandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard let ciImage = CIImage(mtlTexture: outputTexture) else { return nil }
    let newCiimage = ciImage.transformed(by: ciImage.orientationTransform(for: .downMirrored))
    return UIImage(ciImage: newCiimage)

  }

  func makeTextureFromCVPixelBuffer(pixelBuffer: CVPixelBuffer, textureFormat: MTLPixelFormat) -> MTLTexture? {
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)

    // Create a Metal texture from the image buffer.
    var cvTextureOut: CVMetalTexture?
    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, nil, textureFormat, width, height, 0, &cvTextureOut)

    guard let cvTexture = cvTextureOut, let texture = CVMetalTextureGetTexture(cvTexture) else {
      CVMetalTextureCacheFlush(textureCache, 0)
      return nil
    }
    return texture
  }
}

func allocateOutputBufferPool(with imageSize: CGSize, outputRetainedBufferCountHint: Int) ->(
  outputBufferPool: CVPixelBufferPool?,
  outputColorSpace: CGColorSpace?,
  outputFormatDescription: CMFormatDescription?) {

    let width = imageSize.width
    let height = imageSize.height
    /*
     public var kCVPixelFormatType_32ARGB: OSType { get } /* 32 bit ARGB */
     public var kCVPixelFormatType_32ABGR: OSType { get } /* 32 bit ABGR */
     public var kCVPixelFormatType_32RGBA: OSType { get } /* 32 bit RGBA */
     */
    let pixelBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: UInt(kCVPixelFormatType_32ARGB),
      kCVPixelBufferWidthKey as String: Int(width),
      kCVPixelBufferHeightKey as String: Int(height),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    ]

    // Get pixel buffer attributes and color space from the input format description.
    let cgColorSpace = CGColorSpaceCreateDeviceRGB()

    // Create a pixel buffer pool with the same pixel attributes as the input format description.
    let poolAttributes = [kCVPixelBufferPoolMinimumBufferCountKey as String: outputRetainedBufferCountHint]
    var cvPixelBufferPool: CVPixelBufferPool?
    CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as NSDictionary?, pixelBufferAttributes as NSDictionary?, &cvPixelBufferPool)
    guard let pixelBufferPool = cvPixelBufferPool else {
      assertionFailure("Allocation failure: Could not allocate pixel buffer pool.")
      return (nil, nil, nil)
    }

    preallocateBuffers(pool: pixelBufferPool, allocationThreshold: outputRetainedBufferCountHint)

    // Get the output format description.
    var pixelBuffer: CVPixelBuffer?
    var outputFormatDescription: CMFormatDescription?
    let auxAttributes = [kCVPixelBufferPoolAllocationThresholdKey as String: outputRetainedBufferCountHint] as NSDictionary
    CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pixelBufferPool, auxAttributes, &pixelBuffer)
    if let pixelBuffer = pixelBuffer {
      CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                   imageBuffer: pixelBuffer,
                                                   formatDescriptionOut: &outputFormatDescription)
    }
    pixelBuffer = nil

    return (pixelBufferPool, cgColorSpace, outputFormatDescription)
  }

func allocateOutputBufferPool(with inputFormatDescription: CMFormatDescription, outputRetainedBufferCountHint: Int, needChangeWidthHeight: Bool) ->(
  outputBufferPool: CVPixelBufferPool?,
  outputColorSpace: CGColorSpace?,
  outputFormatDescription: CMFormatDescription?) {

    let inputMediaSubType = CMFormatDescriptionGetMediaSubType(inputFormatDescription)
    if inputMediaSubType != kCVPixelFormatType_32BGRA {
      assertionFailure("Invalid input pixel buffer type \(inputMediaSubType)")
      return (nil, nil, nil)
    }

    let inputDimensions = CMVideoFormatDescriptionGetDimensions(inputFormatDescription)
    var width = inputDimensions.width
    var height = inputDimensions.height
    if needChangeWidthHeight {
      width = height
      height = inputDimensions.width
    }
    var pixelBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: UInt(inputMediaSubType),
      kCVPixelBufferWidthKey as String: Int(width),
      kCVPixelBufferHeightKey as String: Int(height),
      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    ]

    // Get pixel buffer attributes and color space from the input format description.
    var cgColorSpace = CGColorSpaceCreateDeviceRGB()
    if let inputFormatDescriptionExtension = CMFormatDescriptionGetExtensions(inputFormatDescription) as Dictionary? {
      let colorPrimaries = inputFormatDescriptionExtension[kCVImageBufferColorPrimariesKey]

      if let colorPrimaries = colorPrimaries {
        var colorSpaceProperties: [String: AnyObject] = [kCVImageBufferColorPrimariesKey as String: colorPrimaries]

        if let yCbCrMatrix = inputFormatDescriptionExtension[kCVImageBufferYCbCrMatrixKey] {
          colorSpaceProperties[kCVImageBufferYCbCrMatrixKey as String] = yCbCrMatrix
        }

        if let transferFunction = inputFormatDescriptionExtension[kCVImageBufferTransferFunctionKey] {
          colorSpaceProperties[kCVImageBufferTransferFunctionKey as String] = transferFunction
        }

        pixelBufferAttributes[kCVBufferPropagatedAttachmentsKey as String] = colorSpaceProperties
      }

      if let cvColorspace = inputFormatDescriptionExtension[kCVImageBufferCGColorSpaceKey] {
        cgColorSpace = cvColorspace as! CGColorSpace
      } else if (colorPrimaries as? String) == (kCVImageBufferColorPrimaries_P3_D65 as String) {
        cgColorSpace = CGColorSpace(name: CGColorSpace.displayP3)!
      }
    }

    // Create a pixel buffer pool with the same pixel attributes as the input format description.
    let poolAttributes = [kCVPixelBufferPoolMinimumBufferCountKey as String: outputRetainedBufferCountHint]
    var cvPixelBufferPool: CVPixelBufferPool?
    CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as NSDictionary?, pixelBufferAttributes as NSDictionary?, &cvPixelBufferPool)
    guard let pixelBufferPool = cvPixelBufferPool else {
      assertionFailure("Allocation failure: Could not allocate pixel buffer pool.")
      return (nil, nil, nil)
    }

    preallocateBuffers(pool: pixelBufferPool, allocationThreshold: outputRetainedBufferCountHint)

    // Get the output format description.
    var pixelBuffer: CVPixelBuffer?
    var outputFormatDescription: CMFormatDescription?
    let auxAttributes = [kCVPixelBufferPoolAllocationThresholdKey as String: outputRetainedBufferCountHint] as NSDictionary
    CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pixelBufferPool, auxAttributes, &pixelBuffer)
    if let pixelBuffer = pixelBuffer {
      CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                   imageBuffer: pixelBuffer,
                                                   formatDescriptionOut: &outputFormatDescription)
    }
    pixelBuffer = nil

    return (pixelBufferPool, cgColorSpace, outputFormatDescription)
  }

/// - Tag: AllocateRenderBuffers
private func preallocateBuffers(pool: CVPixelBufferPool, allocationThreshold: Int) {
  var pixelBuffers = [CVPixelBuffer]()
  var error: CVReturn = kCVReturnSuccess
  let auxAttributes = [kCVPixelBufferPoolAllocationThresholdKey as String: allocationThreshold] as NSDictionary
  var pixelBuffer: CVPixelBuffer?
  while error == kCVReturnSuccess {
    error = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, auxAttributes, &pixelBuffer)
    if let pixelBuffer = pixelBuffer {
      pixelBuffers.append(pixelBuffer)
    }
    pixelBuffer = nil
  }
  pixelBuffers.removeAll()
}

extension UIImage {


func fixedOrientation() -> CGImage? {

    guard let cgImage = self.cgImage else {
      //CGImage is not available
      return nil
    }

    guard imageOrientation != UIImage.Orientation.up else {
      //This is default orientation, don't need to do anything
      return cgImage.copy()
    }

    guard let colorSpace = cgImage.colorSpace, let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
      return nil //Not able to create CGContext
    }
    print(colorSpace)
    var transform: CGAffineTransform = CGAffineTransform.identity

    switch imageOrientation {
    case .down, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: size.height)
      transform = transform.rotated(by: CGFloat.pi)
      break
    case .left, .leftMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.rotated(by: CGFloat.pi / 2.0)
      break
    case .right, .rightMirrored:
      transform = transform.translatedBy(x: 0, y: size.height)
      transform = transform.rotated(by: CGFloat.pi / -2.0)
      break
    default:
      break
    }

    //Flip image one more time if needed to, this is to prevent flipped image
    switch imageOrientation {
    case .upMirrored, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
      break
    case .leftMirrored, .rightMirrored:
      transform = transform.translatedBy(x: size.height, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
    default:
      break
    }

    ctx.concatenate(transform)

    switch imageOrientation {
    case .left, .leftMirrored, .right, .rightMirrored:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.height, height: size.width))
    default:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
      break
    }

    guard let newCGImage = ctx.makeImage() else { return nil }
    return newCGImage
  }
}
