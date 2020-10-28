//
//  Shaders.metal
//  ImageSegmentationAndPose
//
//  Created by Dennis Ippel on 26/10/2020.
//

#include <metal_stdlib>
using namespace metal;

struct ScreenQuadVertex
{
    float4 position [[position]];
    float2 texcoord;
};

struct Uniforms {
    float capturedImageAspectRatio;
    uint classificationLabelIndex;
};

constexpr sampler s = sampler(coord::normalized,
                              address::clamp_to_edge,
                              filter::linear);

constant float4x4 quadVertices = float4x4(float4( -1.0, -1.0, 0.0, 1.0 ),
                                       float4(  1.0, -1.0, 0.0, 1.0 ),
                                       float4( -1.0,  1.0, 0.0, 1.0 ),
                                       float4(  1.0,  1.0, 0.0, 1.0 ));

constant float4x2 quadTextureCoordinates = float4x2(float2( 0.0, 1.0 ),
                                                 float2( 1.0, 1.0 ),
                                                 float2( 0.0, 0.0 ),
                                                 float2( 1.0, 0.0 ));

vertex ScreenQuadVertex screenQuadVertex(uint vertex_id [[vertex_id]],
                                         const constant Uniforms& uniforms [[buffer(0)]])
{
    ScreenQuadVertex outVertex;
    outVertex.position = quadVertices[vertex_id];
    outVertex.position.x *= uniforms.capturedImageAspectRatio;
    outVertex.texcoord = quadTextureCoordinates[vertex_id];
    
    return outVertex;
}

fragment float4 screenQuadFragment(ScreenQuadVertex inVertex [[stage_in]],
                                   float4 frameBufferColor [[color(0)]],
                                   texture2d<uint, access::sample> segmentationTexture [[texture(0)]],
                                   const constant Uniforms& uniforms [[buffer(0)]])
{
    float4 outColor = frameBufferColor;
    float2 uv = inVertex.texcoord;
    uv.x = 1.0 - uv.x;
    uint4 texColor = segmentationTexture.sample(s, uv);
    if(texColor.r == uniforms.classificationLabelIndex) {
        outColor = mix(float4(0, 0, 1, 1), outColor, 0.25);
    }

    return outColor;
}
