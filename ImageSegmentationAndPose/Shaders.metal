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
    float4x4 modelViewProjectionMatrix;
    float capturedImageAspectRatio;
    float nonLinearDepth;
    float2 regionOfInterestOrigin;
    float2 regionOfInterestSize;
    uint classificationLabelIndex;
};

constexpr sampler s = sampler(coord::normalized,
                              address::clamp_to_zero,
                              filter::linear);

constant float4x4 quadVertices = float4x4(float4( -1.0, -1.0, 0.0, 1.0 ),
                                          float4(  1.0, -1.0, 0.0, 1.0 ),
                                          float4( -1.0,  1.0, 0.0, 1.0 ),
                                          float4(  1.0,  1.0, 0.0, 1.0 ));

constant float4x2 quadTextureCoordinates = float4x2(float2( 1.0, 1.0 ),
                                                    float2( 0.0, 1.0 ),
                                                    float2( 1.0, 0.0 ),
                                                    float2( 0.0, 0.0 ));

vertex ScreenQuadVertex screenQuadVertex(uint vertex_id [[vertex_id]],
                                         const constant Uniforms& uniforms [[buffer(0)]])
{
    ScreenQuadVertex outVertex;
    outVertex.position = quadVertices[vertex_id];
    outVertex.position.x *= uniforms.capturedImageAspectRatio;
    outVertex.texcoord = quadTextureCoordinates[vertex_id];
    
    return outVertex;
}

struct FragmentOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment FragmentOut screenQuadFragment(ScreenQuadVertex inVertex [[stage_in]],
                                   float4 frameBufferColor [[color(0)]],
                                   texture2d<uint, access::sample> segmentationTexture [[texture(0)]],
                                   const constant Uniforms& uniforms [[buffer(0)]])
{
    FragmentOut out;
    out.color = float4(0);
    float2 uv = inVertex.texcoord;
    uv.x -= uniforms.regionOfInterestOrigin.x;
    uv.y -= 1.0 - (uniforms.regionOfInterestOrigin.y + uniforms.regionOfInterestSize.y);
    uv /= uniforms.regionOfInterestSize;
    
    uint4 texColor = segmentationTexture.sample(s, uv);
    if(texColor.r == uniforms.classificationLabelIndex) {
        out.color = mix(float4(0, 1, 0, 1), frameBufferColor, 0);
        out.depth = uniforms.nonLinearDepth;
    } else {
        discard_fragment();
    }

    return out;
}
