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

#include <metal_stdlib>
using namespace metal;

kernel void mergeColor(texture2d<half, access::read> inTexture [[ texture (0) ]],
                       texture2d<half, access::read_write> outTexture [[ texture (1) ]],
                       device float* data_in [[ buffer(0) ]],
                       constant int& width [[buffer(1)]],
                       constant int& height [[buffer(2)]],
                       uint2 gid [[ thread_position_in_grid ]]) {
  int scaleWidth = inTexture.get_width();
  int scaleHeight = inTexture.get_height();
  int originX = 0;
  int originY = 0;
  if (scaleWidth > scaleHeight) {
    originY = (scaleWidth - scaleHeight)/2;
    scaleHeight = scaleWidth;
  } else {
    originX = (scaleHeight - scaleWidth)/ 2;
    scaleWidth = scaleHeight;
  }
  uint2 newPoint = uint2(uint((gid.x + originX) * width / scaleWidth),uint((gid.y + originY) * height / scaleHeight));
  half4 pixelData = inTexture.read(gid).rgba;
  float maskValue = data_in[uint(newPoint.y * height  + newPoint.x)];
  half4 outputPixelData = {pixelData.r, pixelData.g, pixelData.b, (pixelData.a * static_cast<half>(maskValue))};
  outTexture.write(outputPixelData, gid);
}
