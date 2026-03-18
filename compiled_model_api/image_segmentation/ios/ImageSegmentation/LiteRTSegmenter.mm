/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#import "LiteRTSegmenter.h"
#import <Accelerate/Accelerate.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"

@implementation LiteRTSegmentationResult
@end

static NSString *const kLiteRTErrorDomain = @"com.google.litert.segmentation";

struct ColoredLabel {
    int r, g, b;
};

static std::vector<ColoredLabel> GetColors() {
    return {
        {0, 0, 0},         // Background
        {255, 0, 0},       // Class 1
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
        auto env_res = litert::Environment::Create({});
        if (!env_res) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:1
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to create LiteRT Environment"}];
            return nil;
        }
        _env = std::move(*env_res);

        auto litert_accel = (accelerator == LiteRTAcceleratorMetal)
            ? litert::HwAccelerators::kGpu
            : litert::HwAccelerators::kCpu;

        auto model_res = litert::CompiledModel::Create(
            *_env, [modelPath UTF8String], litert_accel);

        if (!model_res) {
            NSString *detail = [NSString stringWithFormat:@"Failed to create LiteRT CompiledModel (status=%d): %s",
                static_cast<int>(model_res.Error().Status()),
                model_res.Error().Message().data()];
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:2
                userInfo:@{NSLocalizedDescriptionKey: detail}];
            return nil;
        }
        _model = std::move(*model_res);

        auto inputs_res = _model->CreateInputBuffers();
        if (!inputs_res) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:3
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to create input buffers"}];
            return nil;
        }
        _input_buffers = std::move(*inputs_res);

        auto outputs_res = _model->CreateOutputBuffers();
        if (!outputs_res) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:4
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to create output buffers"}];
            return nil;
        }
        _output_buffers = std::move(*outputs_res);
    }
    return self;
}

- (LiteRTSegmentationResult *)segmentImage:(CGImageRef)image error:(NSError **)error {
    LiteRTSegmentationResult *result = [[LiteRTSegmentationResult alloc] init];

    // 1. Preprocess: resize to model input dimensions and normalize
    auto start_pre = std::chrono::high_resolution_clock::now();

    int modelWidth = 256;
    int modelHeight = 256;
    int channels = 3;

    const size_t bytesPerPixel = 4;
    const size_t bytesPerRow = bytesPerPixel * modelWidth;
    std::vector<uint8_t> imageData(modelHeight * bytesPerRow);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        imageData.data(), modelWidth, modelHeight, 8,
        bytesPerRow, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);

    if (!context) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:5
            userInfo:@{NSLocalizedDescriptionKey: @"Failed to create CGContext for preprocessing"}];
        return nil;
    }

    CGContextDrawImage(context, CGRectMake(0, 0, modelWidth, modelHeight), image);
    CGContextRelease(context);

    // Normalize pixel values to [-1, 1]
    std::vector<float> inputFloat(modelWidth * modelHeight * channels);
    for (int i = 0; i < modelWidth * modelHeight; ++i) {
        uint8_t r = imageData[i * 4 + 0];
        uint8_t g = imageData[i * 4 + 1];
        uint8_t b = imageData[i * 4 + 2];

        inputFloat[i * 3 + 0] = (float(r) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 1] = (float(g) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 2] = (float(b) - 127.5f) / 127.5f;
    }

    // Write to input buffer
    auto input_lock = _input_buffers[0].Lock(litert::TensorBuffer::LockMode::kWrite);
    if (!input_lock) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:7
            userInfo:@{NSLocalizedDescriptionKey: @"Failed to lock input buffer"}];
        return nil;
    }
    memcpy(*input_lock, inputFloat.data(), inputFloat.size() * sizeof(float));
    _input_buffers[0].Unlock();

    auto end_pre = std::chrono::high_resolution_clock::now();
    result.preProcessTimeMs = std::chrono::duration<double, std::milli>(end_pre - start_pre).count();

    // 2. Inference
    auto start_inf = std::chrono::high_resolution_clock::now();
    auto run_status = _model->Run(_input_buffers, _output_buffers);
    if (!run_status) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:6
            userInfo:@{NSLocalizedDescriptionKey: @"Model inference failed"}];
        return nil;
    }
    auto end_inf = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(end_inf - start_inf).count();

    // 3. Postprocess: argmax across channels, map to colored mask
    auto start_post = std::chrono::high_resolution_clock::now();
    auto output_lock = _output_buffers[0].Lock(litert::TensorBuffer::LockMode::kRead);
    if (!output_lock) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:8
            userInfo:@{NSLocalizedDescriptionKey: @"Failed to lock output buffer"}];
        return nil;
    }
    float* out_f = static_cast<float*>(*output_lock);

    int outChannels = 6;

    // Allocate RGBA output owned by NSData to avoid memory leak
    NSMutableData *rgbaData = [NSMutableData dataWithLength:modelWidth * modelHeight * 4];
    uint8_t *outRgba = static_cast<uint8_t *>(rgbaData.mutableBytes);

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

            int rgba_idx = (y * modelWidth + x) * 4;
            if (max_cls == 0) {
                outRgba[rgba_idx + 0] = 0;
                outRgba[rgba_idx + 1] = 0;
                outRgba[rgba_idx + 2] = 0;
                outRgba[rgba_idx + 3] = 0;
            } else {
                outRgba[rgba_idx + 0] = _colors[max_cls].r;
                outRgba[rgba_idx + 1] = _colors[max_cls].g;
                outRgba[rgba_idx + 2] = _colors[max_cls].b;
                outRgba[rgba_idx + 3] = 150;
            }
        }
    }
    _output_buffers[0].Unlock();

    // Create CGImage from RGBA data — use NSData-backed provider to handle memory correctly
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)rgbaData);
    CGColorSpaceRef colorSpaceOutput = CGColorSpaceCreateDeviceRGB();
    CGImageRef cgMask = CGImageCreate(
        modelWidth, modelHeight, 8, 32, modelWidth * 4,
        colorSpaceOutput,
        kCGBitmapByteOrderDefault | kCGImageAlphaLast,
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
