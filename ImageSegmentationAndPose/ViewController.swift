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
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        sceneView.delegate = self
        sceneView.debugOptions = [.showFeaturePoints]
        sceneView.usesReverseZ = true
        
        viewportSize = UIScreen.main.bounds.size
        
        let quadNode = GlowOutlineQuadNode(sceneView: sceneView)
        quadNode.renderingOrder = 100
        quadNode.classificationLabelIndex = UInt(labels.firstIndex(of: "bottle") ?? 0)
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
        
        // kCVPixelFormatType_OneComponent8
        let pixelBuffer = observation.pixelBuffer.copyToMetalCompatible()!
        
//        if commandQueue == nil {
//            commandQueue = sceneView.device!.makeCommandQueue()
//
//            let descr = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: 513, height: 513, mipmapped: false)
//            descr.usage = [.shaderWrite, .shaderRead]
//            let tex = sceneView.device?.makeTexture(descriptor: descr)!
//            renderTex = tex
//        }
//
//        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//        let context = CIContext(mtlDevice: sceneView.device!)
//        context.render(ciImage, to: renderTex!, commandBuffer: nil, bounds: CGRect(x: 0, y: 0, width: 513, height: 513), colorSpace: CGColorSpaceCreateDeviceGray())
//
//        quadNode?.segmentationTexture = renderTex
        
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
    
    func renderer(_ renderer: SCNSceneRenderer, willRenderScene scene: SCNScene, atTime time: TimeInterval) {
        // 1440 x 1920  0.75  |  1440 x 1920   0.75
        //  375 x 812   0.46  |  1024 x 1366   0.75
        // 1125 x 2436  0.46  |  2048 x 2732   0.75
        guard let capturedImage = sceneView.session.currentFrame?.capturedImage else { return }
        
        let capturedImageAspectRatio = Float(CVPixelBufferGetHeight(capturedImage)) / Float(CVPixelBufferGetWidth(capturedImage))
        let screenAspectRatio = Float(viewportSize.width / viewportSize.height)
        quadNode?.correctionAspectRatio = capturedImageAspectRatio * (screenAspectRatio / capturedImageAspectRatio)
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: capturedImage,
                                                        orientation: .leftMirrored,
                                                        options: [:])
        
        do {
            try imageRequestHandler.perform([segmentationRequest])
        } catch {
            print("Failed to perform image request.")
        }
    }
}
