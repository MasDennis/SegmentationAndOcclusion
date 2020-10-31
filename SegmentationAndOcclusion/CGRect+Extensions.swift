//
//  CGRect+Extensions.swift
//  SegmentationAndOcclusion
//
//  Created by Dennis Ippel on 30/10/2020.
//

import CoreGraphics

extension CGRect {
    
    func insetByNormalized(d: CGFloat) -> CGRect {
        let insetRect = insetBy(dx: d, dy: d)
        
        let x = max(0, insetRect.minX)
        let y = max(0, insetRect.minY)
        let width = insetRect.maxX > 1.0 ? 1.0 - x : insetRect.maxX - x
        let height = insetRect.maxY > 1.0 ? 1.0 - y : insetRect.maxY - y

        return CGRect(x: x, y: y, width: width, height: height)
    }
}
