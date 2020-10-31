//
//  Shaders.metal
//  SegmentationAndOcclusion
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
    float aspectRatioAdjustment;
    float depthBufferZ;
    float time;
    float4 regionOfInterest;
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
    outVertex.position.x *= uniforms.aspectRatioAdjustment;
    outVertex.texcoord = quadTextureCoordinates[vertex_id];
    
    return outVertex;
}

struct FragmentOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment FragmentOut screenQuadFragment(ScreenQuadVertex inVertex [[stage_in]],
                                        texture2d<uint, access::sample> segmentationTexture [[texture(0)]],
                                        constant Uniforms& uniforms [[buffer(0)]])
{
    FragmentOut out;
    out.color = float4(0);
    
    float2 uv = inVertex.texcoord;
    // Scales the texture coordinates to the region of interest.
    // regionOfInterest is a CGRect (x, y, width, height) stored in a float4 (x, y, z, w).
    uv.x -= uniforms.regionOfInterest.x;
    uv.y -= 1.0 - (uniforms.regionOfInterest.y + uniforms.regionOfInterest.w);
    uv /= uniforms.regionOfInterest.zw;
    
    uint4 texColor = segmentationTexture.sample(s, uv);
    if(texColor.r == uniforms.classificationLabelIndex) {
        float time = (sin(uniforms.time * 5.0) + 1) * 0.5;
        out.color = float4(1, 1, 0, 0.25 * time);
        out.depth = uniforms.depthBufferZ;
    } else {
        // Discard color and depth information.
        discard_fragment();
    }
    
    return out;
}
