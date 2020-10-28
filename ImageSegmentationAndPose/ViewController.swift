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

    @IBOutlet weak var sceneView: ARSCNView!
    
    private var screenSize: CGSize!
    private var textureCache: CVMetalTextureCache?
    
    private let labels = ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]
    
    private weak var quadNode: GlowOutlineQuadNode?
    
    override var shouldAutorotate: Bool { return false }
    
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
        
        screenSize = UIScreen.main.bounds.size
        
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
    
    private func processSegmentations(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Segmentation error: \(error!.localizedDescription)")
            return
        }
        
        guard let observation = request.results?.first as? VNPixelBufferObservation else { return }

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
    
    private func processObjectDetections(for request: VNRequest, error: Error?) {
        guard error == nil else {
            print("Segmentation error: \(error!.localizedDescription)")
            return
        }
        
        guard var objectObservations = request.results as? [VNRecognizedObjectObservation] else { return }
        
        objectObservations.sort {
            return $0.labels.firstIndex(where: { $0.identifier == "car" }) ?? -1 < $1.labels.firstIndex(where: { $0.identifier == "car" }) ?? -1
        }
        
        guard let bestObservation = objectObservations.first,
              let index = bestObservation.labels.firstIndex(where: { $0.identifier == "car" }),
              index < 3
        else { return }
        
        segmentationRequest.regionOfInterest = bestObservation.boundingBox
        quadNode?.regionOfInterest = bestObservation.boundingBox
    }
    
    func renderer(_ renderer: SCNSceneRenderer, willRenderScene scene: SCNScene, atTime time: TimeInterval) {
        guard let capturedImage = sceneView.session.currentFrame?.capturedImage else { return }
        
        let capturedImageAspectRatio = Float(CVPixelBufferGetWidth(capturedImage)) / Float(CVPixelBufferGetHeight(capturedImage))
        let screenAspectRatio = Float(screenSize.height / screenSize.width)
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
