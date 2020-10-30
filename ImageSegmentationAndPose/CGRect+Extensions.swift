//
//  CGRect+Extensions.swift
//  ImageSegmentationAndPose
//
//  Created by Dennis Ippel on 30/10/2020.
//

import CoreGraphics

extension CGRect {
    
    func insetByNormalized(d: CGFloat) -> CGRect {
        let insetRect = insetBy(dx: d, dy: d)
        
        if insetRect.maxX > 1.0 || insetRect.maxY > 1.0 {
            print("billy")
        }
        
        return CGRect(x: max(0, insetRect.minX),
                      y: max(0, insetRect.minY),
                      width: insetRect.maxX > 1.0 ? 1.0 - insetRect.minX : insetRect.width,
                      height: insetRect.maxY > 1.0 ? 1.0 - insetRect.minY : insetRect.height)
    }
}
