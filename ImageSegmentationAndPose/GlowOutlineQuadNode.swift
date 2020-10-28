//
//  GlowOutlineQuadNode.swift
//  ImageSegmentationAndPose
//
//  Created by Dennis Ippel on 26/10/2020.
//

import SceneKit

class GlowOutlineQuadNode: SCNNode {
    private var renderTexturePipelineState: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState!
    
    var segmentationTexture: MTLTexture?
    private var viewSize = CGSize.zero
    
    struct Uniforms {
        let capturedImageAspectRatio: simd_float1
        let classificationLabelIndex: simd_uint1
    }
    
    var correctionAspectRatio: Float = 0
    var classificationLabelIndex: UInt = 0
    
    init(sceneView: SCNView) {
        super.init()

        rendererDelegate = self
        
        let device = sceneView.device!
        let library = device.makeDefaultLibrary()
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = library?.makeFunction(name: "screenQuadVertex")
        pipelineDescriptor.fragmentFunction = library?.makeFunction(name: "screenQuadFragment")
        pipelineDescriptor.colorAttachments[0].pixelFormat = sceneView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = sceneView.depthPixelFormat

        guard let pipeline = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor) else { return }

        self.renderTexturePipelineState = pipeline
        
        let depthStateDesciptor = MTLDepthStencilDescriptor()
        depthStateDesciptor.isDepthWriteEnabled = false
        guard let state = device.makeDepthStencilState(descriptor:depthStateDesciptor) else { return }
        depthState = state
        
        DispatchQueue.main.async { [weak self] in
            self?.viewSize = sceneView.bounds.size
        }
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

extension GlowOutlineQuadNode: SCNNodeRendererDelegate {
    func renderNode(_ node: SCNNode,
                    renderer: SCNRenderer,
                    arguments: [String : Any]) {
        
        guard let renderTexturePipelineState = renderTexturePipelineState,
              let renderCommandEncoder = renderer.currentRenderCommandEncoder,
              let segmentationTexture = segmentationTexture
        else { return }
        
        var uniforms = Uniforms(capturedImageAspectRatio: correctionAspectRatio,
                                classificationLabelIndex: simd_uint1(classificationLabelIndex))
        
        guard let uniformsBuffer = renderer.device?.makeBuffer(bytes: &uniforms,
                                                               length: MemoryLayout<Uniforms>.size,
                                                               options: [])
        else { return }
        
        renderCommandEncoder.setDepthStencilState(depthState)
        renderCommandEncoder.setRenderPipelineState(renderTexturePipelineState)
        renderCommandEncoder.setFragmentTexture(segmentationTexture, index: 0)
        renderCommandEncoder.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
        renderCommandEncoder.setVertexBuffer(uniformsBuffer, offset: 0, index: 0)
        renderCommandEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}

