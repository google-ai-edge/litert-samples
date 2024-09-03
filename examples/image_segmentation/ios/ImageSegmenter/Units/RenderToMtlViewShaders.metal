// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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


// Vertex input/output structure for passing results from vertex shader to fragment shader
struct VertexIO
{
    float4 position [[position]];
    float2 textureCoord [[user(texturecoord)]];
};

// Vertex shader for a textured quad
vertex VertexIO vertexPassThrough(const device packed_float4 *pPosition  [[ buffer(0) ]],
                                  const device packed_float2 *pTexCoords [[ buffer(1) ]],
                                  uint                  vid        [[ vertex_id ]])
{
    VertexIO outVertex;

    outVertex.position = pPosition[vid];
    outVertex.textureCoord = pTexCoords[vid];

    return outVertex;
}

// Fragment shader for a textured quad
fragment half4 fragmentPassThrough(VertexIO         inputFragment [[ stage_in ]],
                                   texture2d<half> inputTexture  [[ texture(0) ]],
                                   sampler         samplr        [[ sampler(0) ]])
{
    return inputTexture.sample(samplr, inputFragment.textureCoord);
}
