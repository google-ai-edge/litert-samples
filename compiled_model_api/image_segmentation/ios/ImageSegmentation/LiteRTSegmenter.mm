#import "LiteRTSegmenter.h"
#import <Accelerate/Accelerate.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_accelerator_options.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"

@implementation LiteRTSegmentationResult
@end

struct ColoredLabel {
    int r, g, b;
};

// Generates simple colors for the 6 output classes
static std::vector<ColoredLabel> GetColors() {
    return {
        {0, 0, 0},         // Background
        {255, 0, 0},       // Class 1 (e.g. skin)
        {0, 255, 0},       // Class 2
        {0, 0, 255},       // Class 3
        {255, 255, 0},     // Class 4
        {0, 255, 255}      // Class 5
    };
}

@interface LiteRTSegmenter () {
    std::optional<litert::Environment> _env;
    std::optional<litert::CompiledModel> _model;
    std::vector<litert::TensorBuffer> _input_buffers;
    std::vector<litert::TensorBuffer> _output_buffers;
    std::vector<ColoredLabel> _colors;
}
@end

@implementation LiteRTSegmenter

- (instancetype)initWithModelPath:(NSString *)modelPath
                      accelerator:(LiteRTAccelerator)accelerator
                            error:(NSError **)error {
    self = [super init];
    if (self) {
        _colors = GetColors();
        auto env_res = litert::Environment::CreateWithOptions({});
        if (!env_res) {
            if (error) *error = [NSError errorWithDomain:@"LiteRT" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Failed to create LiteRT Environment"}];
            return nil;
        }
        _env = std::move(*env_res);

        auto litert_accel = (accelerator == LiteRTAcceleratorMetal) ? litert::Accelerator::kMetal : litert::Accelerator::kCpu;

        auto model_res = litert::CompiledModel::Create(
            *_env, [modelPath UTF8String],
            litert::CompiledModel::CreateOptions{.hardware_accelerators = {litert_accel}});

        if (!model_res) {
            if (error) *error = [NSError errorWithDomain:@"LiteRT" code:2 userInfo:@{NSLocalizedDescriptionKey: @"Failed to create LiteRT CompiledModel"}];
            return nil;
        }
        _model = std::move(*model_res);

        auto inputs_res = _model->CreateInputBuffers();
        if (!inputs_res) return nil;
        _input_buffers = std::move(*inputs_res);

        auto outputs_res = _model->CreateOutputBuffers();
        if (!outputs_res) return nil;
        _output_buffers = std::move(*outputs_res);
    }
    return self;
}

- (LiteRTSegmentationResult *)segmentImage:(CGImageRef)image error:(NSError **)error {
    LiteRTSegmentationResult *result = [[LiteRTSegmentationResult alloc] init];
    
    // 1. Preprocess
    auto start_pre = std::chrono::high_resolution_clock::now();
    
    int modelWidth = 256;
    int modelHeight = 256;
    int channels = 3;
    
    // Create an ARGB CoreGraphics context to manually draw and resize the image
    const size_t bytesPerPixel = 4;
    const size_t bytesPerRow = bytesPerPixel * modelWidth;
    std::vector<uint8_t> imageData(modelHeight * bytesPerRow);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(imageData.data(), modelWidth, modelHeight, 8,
                                                 bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    if (!context) {
        if (error) *error = [NSError errorWithDomain:@"LiteRT" code:3 userInfo:@{NSLocalizedDescriptionKey: @"Failed to create CGContext."}];
        return nil;
    }
    
    CGContextDrawImage(context, CGRectMake(0, 0, modelWidth, modelHeight), image);
    CGContextRelease(context);
    
    // Normalize to float: (val - 127.5) / 127.5
    std::vector<float> inputFloat(modelWidth * modelHeight * channels);
    for (int i = 0; i < modelWidth * modelHeight; ++i) {
        uint8_t r = imageData[i * 4 + 0];
        uint8_t g = imageData[i * 4 + 1];
        uint8_t b = imageData[i * 4 + 2];
        
        inputFloat[i * 3 + 0] = (float(r) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 1] = (float(g) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 2] = (float(b) - 127.5f) / 127.5f;
    }
    
    // Write Float buffer
    void* input_ptr;
    _input_buffers[0].Lock(&input_ptr);
    memcpy(input_ptr, inputFloat.data(), inputFloat.size() * sizeof(float));
    _input_buffers[0].Unlock();
    
    auto end_pre = std::chrono::high_resolution_clock::now();
    result.preProcessTimeMs = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();
    
    // 2. Inference
    auto start_inf = std::chrono::high_resolution_clock::now();
    auto run_status = _model->Run(_input_buffers, _output_buffers);
    if (!run_status) {
        if (error) *error = [NSError errorWithDomain:@"LiteRT" code:4 userInfo:@{NSLocalizedDescriptionKey: @"Model Run Failed"}];
        return nil;
    }
    auto end_inf = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(end_inf - start_inf).count();
    
    // 3. Postprocess
    auto start_post = std::chrono::high_resolution_clock::now();
    void* output_ptr;
    _output_buffers[0].Lock(&output_ptr);
    float* out_f = static_cast<float*>(output_ptr);
    
    // Assuming output is 1 x 256 x 256 x 6
    int outChannels = 6;
    std::vector<uint8_t> outRgba(modelWidth * modelHeight * 4);
    
    for (int y = 0; y < modelHeight; ++y) {
        for (int x = 0; x < modelWidth; ++x) {
            int pixel_base = (y * modelWidth + x) * outChannels;
            float max_val = out_f[pixel_base];
            int max_cls = 0;
            for (int c = 1; c < outChannels; ++c) {
                if (out_f[pixel_base + c] > max_val) {
                    max_val = out_f[pixel_base + c];
                    max_cls = c;
                }
            }
            
            // Map to RGBA
            int rgba_idx = (y * modelWidth + x) * 4;
            if (max_cls == 0) { // Background mask transparent
                outRgba[rgba_idx + 0] = 0;
                outRgba[rgba_idx + 1] = 0;
                outRgba[rgba_idx + 2] = 0;
                outRgba[rgba_idx + 3] = 0;
            } else { // Label coloring with semitransparent alpha
                outRgba[rgba_idx + 0] = _colors[max_cls].r;
                outRgba[rgba_idx + 1] = _colors[max_cls].g;
                outRgba[rgba_idx + 2] = _colors[max_cls].b;
                outRgba[rgba_idx + 3] = 150; // alpha
            }
        }
    }
    _output_buffers[0].Unlock();
    
    // Create CGImage from RGBA
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, outRgba.data(), outRgba.size(), NULL);
    CGColorSpaceRef colorSpaceOutput = CGColorSpaceCreateDeviceRGB();
    CGImageRef cgMask = CGImageCreate(modelWidth, modelHeight, 8, 32, modelWidth * 4,
                                      colorSpaceOutput, kCGBitmapByteOrderDefault | kCGImageAlphaLast,
                                      provider, NULL, NO, kCGRenderingIntentDefault);
    
    result.maskImage = [UIImage imageWithCGImage:cgMask];
    
    CGImageRelease(cgMask);
    CGColorSpaceRelease(colorSpaceOutput);
    CGDataProviderRelease(provider);
    
    auto end_post = std::chrono::high_resolution_clock::now();
    result.postProcessTimeMs = std::chrono::duration<double, std::milli>(end_post - start_post).count();
    
    return result;
}

@end
