//
//  HighlighterNode.swift
//  SegmentationAndOcclusion
//
//  Created by Dennis Ippel on 31/10/2020.
//

import SceneKit

class HighlighterNode: SCNNode {
    
    override init() {
        super.init()
        setupNode()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupNode() {
        let outerRing = SCNNode(geometry: SCNTorus(ringRadius: 0.125, pipeRadius: 0.005))
        outerRing.geometry?.firstMaterial?.lightingModel = .physicallyBased
        outerRing.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        outerRing.geometry?.firstMaterial?.metalness.contents = 1.0
        outerRing.geometry?.firstMaterial?.roughness.contents = 0.0
        addChildNode(outerRing)
        
        let outerRingAction = SCNAction.repeatForever(SCNAction.rotateBy(x: .pi * 2.0, y: 0, z: 0, duration: 6.0))
        outerRing.runAction(outerRingAction)

        let innerRing = SCNNode(geometry: SCNTorus(ringRadius: 0.115, pipeRadius: 0.0045))
        innerRing.geometry?.firstMaterial?.lightingModel = .physicallyBased
        innerRing.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        innerRing.geometry?.firstMaterial?.metalness.contents = 1.0
        innerRing.geometry?.firstMaterial?.roughness.contents = 0.0
        addChildNode(innerRing)
        
        let innerRingAction = SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: 0, z: .pi * 2.0, duration: 4.0))
        innerRing.runAction(innerRingAction)
    }
}
