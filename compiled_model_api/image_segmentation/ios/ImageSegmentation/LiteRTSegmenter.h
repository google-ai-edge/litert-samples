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
