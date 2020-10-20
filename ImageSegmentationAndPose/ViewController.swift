//
//  ViewController.swift
//  ImageSegmentationAndPose
//
//  Created by Dennis Ippel on 07/10/2020.
//

import UIKit
import Vision
import ARKit
import Accelerate
import CoreImage

class ViewController: UIViewController, ARSCNViewDelegate {

    @IBOutlet var sceneView: ARSCNView!
    
    private var viewportSize: CGSize!
    
    override var shouldAutorotate: Bool { return false }
    
    lazy var segmentationRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: DeepLabV3Int8LUT(configuration: MLModelConfiguration()).model)
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.processSegmentations(for: request, error: error)
            }
            return request
        } catch {
            fatalError("Failed to load Vision ML model.")
        }
    }()
    
    lazy var objectDetectionRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: YOLOv3TinyInt8LUT(configuration: MLModelConfiguration()).model)
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.processObjectDetections(for: request, error: error)
            }
            return request
        } catch {
            fatalError("Failed to load Vision ML model.")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        sceneView.delegate = self
        
        viewportSize = sceneView.frame.size
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        resetTracking()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }
    
    private func resetTracking() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = []
        sceneView.session.run(configuration, options: [.removeExistingAnchors, .resetTracking])
    }
    
    var recognizedObjectBoundingBox: CGRect?
    
    private func processObjectDetections(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Object detection error: \(error!.localizedDescription)")
            return
        }
        print("processdetections")
        guard let objectObservation = request.results?.first as? VNRecognizedObjectObservation,
              let label = objectObservation.labels.first,
              label.identifier == "cup",
              label.confidence > 0.9
        else {
            recognizedObjectBoundingBox = nil
            return
        }
        
        recognizedObjectBoundingBox = objectObservation.boundingBox
    }

    
    private func processSegmentations(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Segmentation error: \(error!.localizedDescription)")
            return
        }
        
        print("processsegmentations")

        guard let observation = request.results?.first as? VNCoreMLFeatureValueObservation,
              let boundingBox = recognizedObjectBoundingBox,
              let multiArray = observation.featureValue.multiArrayValue
        else { return }
        
        print("segmen2222")

        
        let mapSize = CGSize(width: multiArray.shape[0].intValue,
                             height: multiArray.shape[1].intValue)
        let segmentationMap = observation.featureValue.multiArrayValue
        print(mapSize)
//        let image = segmentationMap?.cgImage()
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
    }
    
    func renderer(_ renderer: SCNSceneRenderer, willRenderScene scene: SCNScene, atTime time: TimeInterval) {
        guard let capturedImage = sceneView.session.currentFrame?.capturedImage else { return }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: capturedImage,
                                                        orientation: .leftMirrored,
                                                        options: [:])
        
        do {
            try imageRequestHandler.perform([objectDetectionRequest, segmentationRequest])
        } catch {
            print("Failed to perform image request.")
        }
    }
}
