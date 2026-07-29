#import "LiteRTSegmenter.h"
#import <Accelerate/Accelerate.h>
#include <dlfcn.h>

#include <vector>
#include <chrono>
#include <cstring>

#include "litert_common.h"
#include "litert_environment.h"
#include "litert_model.h"
#include "litert_options.h"
#include "litert_compiled_model.h"
#include "litert_tensor_buffer.h"
#include "litert_tensor_buffer_requirements.h"
#include "litert_opaque_options.h"

// Declare internal APIs for manual GPU registration and CPU option configuration
namespace litert {
namespace internal {
LiteRtStatus RegisterGpuAccelerator(LiteRtEnvironment environment);
}
}

typedef struct LrtCpuOptions LrtCpuOptions;

typedef enum {
  kLiteRtCpuKernelModeXnnpack = 0,
  kLiteRtCpuKernelModeReference = 1,
  kLiteRtCpuKernelModeBuiltin = 2,
} LiteRtCpuKernelMode;

extern "C" {
LiteRtStatus LrtCreateCpuOptions(LrtCpuOptions** options);
void LrtDestroyCpuOptions(LrtCpuOptions* options);
LiteRtStatus LrtSetCpuOptionsKernelMode(LrtCpuOptions* options, LiteRtCpuKernelMode mode);
LiteRtStatus LrtGetOpaqueCpuOptionsData(const LrtCpuOptions* options,
                                        const char** identifier,
                                        const void** payload,
                                        void (**payload_deleter)(void*));
}

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
    LiteRtEnvironment _env;
    LiteRtModel _model_ref;
    LiteRtCompiledModel _model;
    LiteRtTensorBuffer _input_buffer;
    LiteRtTensorBuffer _output_buffer;
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
        _env = nullptr;
        _model_ref = nullptr;
        _model = nullptr;
        _input_buffer = nullptr;
        _output_buffer = nullptr;

        // 1. Create LiteRT Environment with the private frameworks path as runtime library directory
        NSString *frameworksPath = [[NSBundle mainBundle] privateFrameworksPath];
        LiteRtEnvOption env_options[1];
        env_options[0].tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
        env_options[0].value.type = kLiteRtAnyTypeString;
        env_options[0].value.str_value = [frameworksPath UTF8String];

        LiteRtStatus status = LiteRtCreateEnvironment(1, env_options, &_env);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:1
                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed to create LiteRT Environment (status=%d)", status]}];
            return nil;
        }

        // Diagnostics: Manually call dlopen to retrieve exact system error if loading fails
        {
            NSString *dylibPath = [frameworksPath stringByAppendingPathComponent:@"libLiteRtMetalAccelerator.dylib"];
            void* test_handle = dlopen([dylibPath UTF8String], RTLD_LAZY | RTLD_LOCAL);
            if (!test_handle) {
                NSLog(@"[LiteRTSegmenter] Diagnostics: Manual dlopen of Metal dylib failed: %s", dlerror());
            } else {
                NSLog(@"[LiteRTSegmenter] Diagnostics: Manual dlopen of Metal dylib succeeded!");
                dlclose(test_handle);
            }
        }

        // 2. Manually trigger GPU accelerator registration (will dynamic load libLiteRtMetalAccelerator.dylib from frameworks path)
        LiteRtStatus gpu_reg_status = litert::internal::RegisterGpuAccelerator(_env);
        if (gpu_reg_status != kLiteRtStatusOk) {
            NSLog(@"[LiteRTSegmenter] GPU accelerator registration failed with status: %d", gpu_reg_status);
        } else {
            NSLog(@"[LiteRTSegmenter] GPU accelerator registration succeeded.");
        }

        // 3. Create LiteRtModel from file path
        status = LiteRtCreateModelFromFile(_env, [modelPath UTF8String], &_model_ref);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:2
                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed to load model file (status=%d)", status]}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        // 4. Create Compilation Options
        LiteRtOptions options = nullptr;
        status = LiteRtCreateOptions(&options);
        if (status == kLiteRtStatusOk) {
            LiteRtHwAcceleratorSet accel_set = (accelerator == LiteRTAcceleratorMetal)
                ? (kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu)
                : kLiteRtHwAcceleratorCpu;
            LiteRtSetOptionsHardwareAccelerators(options, accel_set);

            // Configure CPU options to use Builtin kernels instead of XNNPACK (applies to CPU standalone & GPU CPU fallback)
            LrtCpuOptions* cpu_opts = nullptr;
            if (LrtCreateCpuOptions(&cpu_opts) == kLiteRtStatusOk) {
                LrtSetCpuOptionsKernelMode(cpu_opts, kLiteRtCpuKernelModeBuiltin);
                const char* identifier = nullptr;
                const void* payload = nullptr;
                void (*payload_deleter)(void*) = nullptr;
                if (LrtGetOpaqueCpuOptionsData(cpu_opts, &identifier, &payload, &payload_deleter) == kLiteRtStatusOk) {
                    LiteRtOpaqueOptions opaque_opts = nullptr;
                    if (LiteRtCreateOpaqueOptions(identifier, const_cast<void*>(payload), payload_deleter, &opaque_opts) == kLiteRtStatusOk) {
                        LiteRtAddOpaqueOptions(options, opaque_opts);
                    }
                }
            }
        }

        // 5. Create Compiled Model
        status = LiteRtCreateCompiledModel(_env, _model_ref, options, &_model);
        if (options) LiteRtDestroyOptions(options);

        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:3
                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed to compile model (status=%d)", status]}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        // 6. Query layout and allocate input buffer using dynamic requirements
        LiteRtLayout input_layout;
        status = LiteRtGetCompiledModelInputTensorLayout(_model, 0, 0, &input_layout);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:4
                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed to query input tensor layout (status=%d)", status]}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        LiteRtRankedTensorType input_tensor_type;
        input_tensor_type.element_type = kLiteRtElementTypeFloat32;
        input_tensor_type.layout = input_layout;

        LiteRtTensorBufferRequirements input_req = nullptr;
        status = LiteRtGetCompiledModelInputBufferRequirements(_model, 0, 0, &input_req);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:5
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to query input buffer requirements"}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        status = LiteRtCreateManagedTensorBufferFromRequirements(_env, &input_tensor_type, input_req, &_input_buffer);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:6
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to create managed input buffer"}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        // 7. Query layout and allocate output buffer using dynamic requirements
        LiteRtLayout output_layout;
        status = LiteRtGetCompiledModelOutputTensorLayouts(_model, 0, 1, &output_layout, false);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:7
                userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed to query output tensor layout (status=%d)", status]}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        LiteRtRankedTensorType output_tensor_type;
        output_tensor_type.element_type = kLiteRtElementTypeFloat32;
        output_tensor_type.layout = output_layout;

        LiteRtTensorBufferRequirements output_req = nullptr;
        status = LiteRtGetCompiledModelOutputBufferRequirements(_model, 0, 0, &output_req);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:8
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to query output buffer requirements"}];
            [self deallocBuffersAndCleanup];
            return nil;
        }

        status = LiteRtCreateManagedTensorBufferFromRequirements(_env, &output_tensor_type, output_req, &_output_buffer);
        if (status != kLiteRtStatusOk) {
            if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:9
                userInfo:@{NSLocalizedDescriptionKey: @"Failed to create managed output buffer"}];
            [self deallocBuffersAndCleanup];
            return nil;
        }
    }
    return self;
}

- (void)dealloc {
    [self deallocBuffersAndCleanup];
}

- (void)deallocBuffersAndCleanup {
    if (_input_buffer) {
        LiteRtDestroyTensorBuffer(_input_buffer);
        _input_buffer = nullptr;
    }
    if (_output_buffer) {
        LiteRtDestroyTensorBuffer(_output_buffer);
        _output_buffer = nullptr;
    }
    if (_model) {
        LiteRtDestroyCompiledModel(_model);
        _model = nullptr;
    }
    if (_model_ref) {
        LiteRtDestroyModel(_model_ref);
        _model_ref = nullptr;
    }
    if (_env) {
        LiteRtDestroyEnvironment(_env);
        _env = nullptr;
    }
}

- (nullable LiteRTSegmentationResult *)segmentImage:(CGImageRef)image
                                               error:(NSError **)error {
    if (!_model || !_input_buffer || !_output_buffer) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:10
            userInfo:@{NSLocalizedDescriptionKey: @"Segmenter is not initialized"}];
        return nil;
    }

    // 1. Get image dimension details
    const int modelWidth = 256;
    const int modelHeight = 256;
    const int channels = 3;

    // 2. Extract image pixel bytes (drawing to standard format 32bpp RGBA)
    std::vector<uint8_t> imageData(modelWidth * modelHeight * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        imageData.data(), modelWidth, modelHeight, 8, modelWidth * 4, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    
    CGContextDrawImage(context, CGRectMake(0, 0, modelWidth, modelHeight), image);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);

    auto start_preprocess = std::chrono::high_resolution_clock::now();

    // 3. Preprocess and normalize pixel values to [-1, 1] (NHWC float32)
    std::vector<float> inputFloat(modelWidth * modelHeight * channels);
    for (int i = 0; i < modelWidth * modelHeight; ++i) {
        uint8_t r = imageData[i * 4 + 0];
        uint8_t g = imageData[i * 4 + 1];
        uint8_t b = imageData[i * 4 + 2];

        inputFloat[i * 3 + 0] = (float(r) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 1] = (float(g) - 127.5f) / 127.5f;
        inputFloat[i * 3 + 2] = (float(b) - 127.5f) / 127.5f;
    }

    auto end_preprocess = std::chrono::high_resolution_clock::now();

    // 4. Lock input tensor buffer and copy float data
    void* host_addr = nullptr;
    LiteRtStatus status = LiteRtLockTensorBuffer(_input_buffer, &host_addr, kLiteRtTensorBufferLockModeWrite);
    if (status != kLiteRtStatusOk) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:11
            userInfo:@{NSLocalizedDescriptionKey: @"Failed to lock input tensor buffer"}];
        return nil;
    }
    memcpy(host_addr, inputFloat.data(), inputFloat.size() * sizeof(float));
    LiteRtUnlockTensorBuffer(_input_buffer);

    auto start_inference = std::chrono::high_resolution_clock::now();

    // 5. Run compiled model inference
    status = LiteRtRunCompiledModel(_model, 0, 1, &_input_buffer, 1, &_output_buffer);
    if (status != kLiteRtStatusOk) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:12
            userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Model inference failed (status=%d)", status]}];
        return nil;
    }

    auto end_inference = std::chrono::high_resolution_clock::now();
    auto start_postprocess = std::chrono::high_resolution_clock::now();

    // 6. Lock output tensor buffer and generate mask image
    const float* outputData = nullptr;
    status = LiteRtLockTensorBuffer(_output_buffer, (void**)&outputData, kLiteRtTensorBufferLockModeRead);
    if (status != kLiteRtStatusOk) {
        if (error) *error = [NSError errorWithDomain:kLiteRTErrorDomain code:13
            userInfo:@{NSLocalizedDescriptionKey: @"Failed to lock output tensor buffer"}];
        return nil;
    }

    // Output is 1 x 256 x 256 x 6
    // Generate RGBA mask bitmap based on the argmax of the 6 classes
    std::vector<uint8_t> maskBytes(modelWidth * modelHeight * 4);
    for (int y = 0; y < modelHeight; ++y) {
        for (int x = 0; x < modelWidth; ++x) {
            int pixelOffset = (y * modelWidth + x);
            int classOffset = pixelOffset * 6;

            int maxClass = 0;
            float maxVal = outputData[classOffset];
            for (int c = 1; c < 6; ++c) {
                if (outputData[classOffset + c] > maxVal) {
                    maxVal = outputData[classOffset + c];
                    maxClass = c;
                }
            }

            ColoredLabel col = _colors[maxClass];
            maskBytes[pixelOffset * 4 + 0] = col.r;
            maskBytes[pixelOffset * 4 + 1] = col.g;
            maskBytes[pixelOffset * 4 + 2] = col.b;
            maskBytes[pixelOffset * 4 + 3] = (maxClass == 0) ? 0 : 180; // transparent for background, 180 alpha for classes
        }
    }
    LiteRtUnlockTensorBuffer(_output_buffer);

    // Create UIImage from raw mask pixels
    CGColorSpaceRef maskColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef maskContext = CGBitmapContextCreate(
        maskBytes.data(), modelWidth, modelHeight, 8, modelWidth * 4, maskColorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    
    CGImageRef maskImageRef = CGBitmapContextCreateImage(maskContext);
    UIImage *maskImage = [UIImage imageWithCGImage:maskImageRef];
    
    CGImageRelease(maskImageRef);
    CGContextRelease(maskContext);
    CGColorSpaceRelease(maskColorSpace);

    auto end_postprocess = std::chrono::high_resolution_clock::now();

    // Create and populate result object
    LiteRTSegmentationResult *result = [[LiteRTSegmentationResult alloc] init];
    result.maskImage = maskImage;
    result.preProcessTimeMs = std::chrono::duration<double, std::milli>(end_preprocess - start_preprocess).count();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();
    result.postProcessTimeMs = std::chrono::duration<double, std::milli>(end_postprocess - start_postprocess).count();

    return result;
}

@end
