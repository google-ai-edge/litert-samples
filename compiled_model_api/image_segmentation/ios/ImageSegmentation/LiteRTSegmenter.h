/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, LiteRTAccelerator) {
    LiteRTAcceleratorCPU = 0,
    LiteRTAcceleratorMetal = 1
};

@interface LiteRTSegmentationResult : NSObject
@property (nonatomic, strong) UIImage *maskImage;
@property (nonatomic, assign) NSTimeInterval preProcessTimeMs;
@property (nonatomic, assign) NSTimeInterval inferenceTimeMs;
@property (nonatomic, assign) NSTimeInterval postProcessTimeMs;
@end

@interface LiteRTSegmenter : NSObject

- (instancetype)init NS_UNAVAILABLE;

/// Initialize the segmenter with the path to the model file and the chosen accelerator.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                               accelerator:(LiteRTAccelerator)accelerator
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/// Segments the given image (CGImage) and returns a visual mask.
- (nullable LiteRTSegmentationResult *)segmentImage:(CGImageRef)image
                                              error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
