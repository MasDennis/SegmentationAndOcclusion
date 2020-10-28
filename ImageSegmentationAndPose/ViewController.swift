//
//  ViewController.swift
//  ImageSegmentationAndPose
//
//  Created by Dennis Ippel on 07/10/2020.
//

// https://machinethink.net/blog/coreml-image-mlmultiarray/
// import coremltools
// import coremltools.proto.FeatureTypes_pb2 as ft
// spec = coremltools.utils.load_spec("DeepLabV3Int8LUT.mlmodel")
// print(spec.description)
// output = spec.description.output[0]
// output.type.imageType.height = 513
// output.type.imageType.width = 513
// output.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
// coremltools.utils.save_spec(spec, "DeepLabV3Int8Image.mlmodel"

import UIKit
import Vision
import ARKit
import Accelerate
import CoreImage

class ViewController: UIViewController, ARSCNViewDelegate {

    @IBOutlet var sceneView: ARSCNView!
    
    private var viewportSize: CGSize!
    
    private let labels = ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]
    
    override var shouldAutorotate: Bool { return false }
    
    private weak var quadNode: GlowOutlineQuadNode?
    
    private var textureCache: CVMetalTextureCache?
    
    lazy var segmentationRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: DeepLabV3Int8Image(configuration: MLModelConfiguration()).model)
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.processSegmentations(for: request, error: error)
            }
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch {
            fatalError("Failed to load Vision ML model.")
        }
    }()
    
    lazy var objectDetectionRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: YOLOv3Int8LUT(configuration: MLModelConfiguration()).model)
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.processObjectDetections(for: request, error: error)
            }
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch {
            fatalError("Failed to load Vision ML model.")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        sceneView.delegate = self
        sceneView.debugOptions = [.showFeaturePoints]
        sceneView.showsStatistics = true
        sceneView.contentScaleFactor = max(1, sceneView.contentScaleFactor - 1)
        
        viewportSize = UIScreen.main.bounds.size
        
        let quadNode = GlowOutlineQuadNode(sceneView: sceneView)
        quadNode.renderingOrder = 100
        quadNode.classificationLabelIndex = UInt(labels.firstIndex(of: "car") ?? 0)
        sceneView.scene.rootNode.addChildNode(quadNode)
        
        self.quadNode = quadNode
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
    
    private var commandQueue: MTLCommandQueue?
    private var renderTex: MTLTexture?
    
    private func processSegmentations(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Segmentation error: \(error!.localizedDescription)")
            return
        }
        
        guard let observation = request.results?.first as? VNPixelBufferObservation else { return }

        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0
        defer { SCNTransaction.commit() }
        
        let pixelBuffer = observation.pixelBuffer.copyToMetalCompatible()!
        
        if textureCache == nil && CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, MTLCreateSystemDefaultDevice()!, nil, &textureCache) != kCVReturnSuccess {
            assertionFailure()
            return
        }

        var segmentationTexture: CVMetalTexture?

        let result = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                               textureCache.unsafelyUnwrapped,
                                                               pixelBuffer,
                                                               nil,
                                                               .r8Uint,
                                                               CVPixelBufferGetWidth(pixelBuffer),
                                                               CVPixelBufferGetHeight(pixelBuffer),
                                                               0,
                                                               &segmentationTexture)

        guard result == kCVReturnSuccess,
            let image = segmentationTexture,
            let texture = CVMetalTextureGetTexture(image)
            else { return }

        quadNode?.segmentationTexture = texture
    }
    
//    var objDetRec: CGRect = .zero
    
    private func processObjectDetections(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Segmentation error: \(error!.localizedDescription)")
            return
        }
        
        guard let results = request.results else { return }
        
        var bestIndex = Int.max
        var bestObs: VNRecognizedObjectObservation?
            for observation in results where observation is VNRecognizedObjectObservation {
                guard let objectObservation = observation as? VNRecognizedObjectObservation,
                    let topLabelObservation = objectObservation.labels.first
                    else { continue }
                
                if let obs = objectObservation.labels.first(where: { $0.identifier == "car" }),
                   let index = objectObservation.labels.firstIndex(of: obs) {
                    
                    if index < 3, index < bestIndex {
                        bestIndex = index
                        bestObs = objectObservation
                    }
                }
                
//                guard topLabelObservation.identifier == "car" else { continue }
                
//                segmentationRequest.regionOfInterest = objectObservation.boundingBox
//                quadNode?.regionOfInterest = objectObservation.boundingBox
//                return
//                objDetRec = objectObservation.boundingBox
//                return
//                guard let capturedImage = sceneView.session.currentFrame?.capturedImage else { return }
//
//                let segmentationRequestHandler = VNImageRequestHandler(cvPixelBuffer: capturedImage,
//                                                                       orientation: .leftMirrored,
//                                                                       options: [:])
//                segmentationRequest.regionOfInterest = objectObservation.boundingBox
//
//                do {
//                    try segmentationRequestHandler.perform([segmentationRequest])
//                } catch {
//                    print("Failed to perform image request.")
//                }
            }
        
        if let best = bestObs {
            segmentationRequest.regionOfInterest = best.boundingBox
            quadNode?.regionOfInterest = best.boundingBox
        }
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, willRenderScene scene: SCNScene, atTime time: TimeInterval) {
        guard let capturedImage = sceneView.session.currentFrame?.capturedImage else { return }
        
        let capturedImageAspectRatio = Float(CVPixelBufferGetWidth(capturedImage)) / Float(CVPixelBufferGetHeight(capturedImage))
        let screenAspectRatio = Float(viewportSize.height / viewportSize.width)
        quadNode?.correctionAspectRatio = screenAspectRatio / capturedImageAspectRatio
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: capturedImage,
                                                        orientation: .leftMirrored,
                                                        options: [:])
        do {
            try imageRequestHandler.perform([objectDetectionRequest, segmentationRequest])
        } catch {
            print("Failed to perform image request. \(error)")
            segmentationRequest.regionOfInterest = CGRect(x: 0, y: 0, width: 1, height: 1)
        }
    }
}
