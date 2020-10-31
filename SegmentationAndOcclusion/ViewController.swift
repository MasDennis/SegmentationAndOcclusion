//
//  ViewController.swift
//  SegmentationAndOcclusion
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

class ViewController: UIViewController, ARSCNViewDelegate {

    @IBOutlet weak var sceneView: ARSCNView!
    
    private var screenSize: CGSize!
    private var textureCache: CVMetalTextureCache?
    
    // Deeplab labels
    private let labels = ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]
    private let targetObjectLabel = "car"
    private var targetObjectLabelindex: UInt { return UInt(labels.firstIndex(of: targetObjectLabel) ?? 0) }
    
    private weak var quadNode: SegmentationMaskNode?
    private weak var highlighterNode: SCNNode?
    
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
        
        let quadNode = SegmentationMaskNode(sceneView: sceneView)
        quadNode.classificationLabelIndex = targetObjectLabelindex
        sceneView.scene.rootNode.addChildNode(quadNode)
        
        self.quadNode = quadNode
        
        let highlighterNode = HighlighterNode()
        highlighterNode.isHidden = true
        sceneView.scene.rootNode.addChildNode(highlighterNode)
        
        self.highlighterNode = highlighterNode
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
        configuration.environmentTexturing = .automatic
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
            print("Object detection error: \(error!.localizedDescription)")
            return
        }
        
        guard var objectObservations = request.results as? [VNRecognizedObjectObservation] else { return }
        
        objectObservations.sort {
            return $0.labels.firstIndex(where: { $0.identifier == targetObjectLabel }) ?? -1 < $1.labels.firstIndex(where: { $0.identifier == targetObjectLabel }) ?? -1
        }
        
        guard let bestObservation = objectObservations.first,
              let index = bestObservation.labels.firstIndex(where: { $0.identifier == targetObjectLabel }),
              index < 3
        else { return }
        
        // Enlarge the region of interest slightly to get better results
        let enlargedRegionOfInterest = bestObservation.boundingBox.insetByNormalized(d: -0.02)

        segmentationRequest.regionOfInterest = enlargedRegionOfInterest
        quadNode?.regionOfInterest = enlargedRegionOfInterest
        placeHighlighterNode(atCenterOfBoundingBox: enlargedRegionOfInterest)
    }
    
    private func placeHighlighterNode(atCenterOfBoundingBox boundingBox: CGRect) {
        guard let displayTransform = sceneView.session.currentFrame?.displayTransform(for: .landscapeLeft, viewportSize: screenSize) else { return }
        
        let viewBoundingBox = viewSpaceBoundingBox(fromNormalizedImageBoundingBox: boundingBox,
                                                   usingDisplayTransform: displayTransform)
        
        let midPoint = CGPoint(x: viewBoundingBox.midX,
                               y: viewBoundingBox.midY)

        let results = sceneView.hitTest(midPoint, types: .featurePoint)
        
        guard let result = results.first else { return }
        
        let translation =  result.worldTransform.columns.3
        highlighterNode?.simdWorldPosition = simd_float3(translation.x, translation.y, translation.z)
        highlighterNode?.isHidden = false
        
        guard let camera = sceneView.pointOfView?.camera else { return }
        
        quadNode?.depthBufferZ = calculateFragmentDepth(usingCamera: camera,
                                                          distanceToTarget: Double(result.distance),
                                                          usesReverseZ: sceneView.usesReverseZ)
    }
    
    private func calculateFragmentDepth(usingCamera camera: SCNCamera, distanceToTarget: Double, usesReverseZ: Bool) -> Float {
        let zFar = usesReverseZ ? camera.zNear : camera.zFar
        let zNear = usesReverseZ ? camera.zFar : camera.zNear
        let range = 2.0 * zNear * zFar
        let fragmentDepth = (zFar + zNear - range / distanceToTarget) / (zFar - zNear)

        return Float((fragmentDepth + 1.0) / 2.0)
    }
    
    private func viewSpaceBoundingBox(fromNormalizedImageBoundingBox imageBoundingBox: CGRect,
                                      usingDisplayTransform displayTransorm: CGAffineTransform) -> CGRect {
        // Transform into normalized view coordinates
        let viewNormalizedBoundingBox = imageBoundingBox.applying(displayTransorm)
        // The affine transform for view coordinates
        let t = CGAffineTransform(scaleX: screenSize.width, y: screenSize.height)
        // Scale up to view coordinates
        return viewNormalizedBoundingBox.applying(t)
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
        }
    }
}
