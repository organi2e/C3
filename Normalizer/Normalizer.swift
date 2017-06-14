//
//  Normalizer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/06/12.
//
//
import Metal
public protocol Normalizer {
	func adjust(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer))
	func flush(commandBuffer: MTLCommandBuffer)
}
public class PassThrough {
	public init(count: Int) {
		
	}
}
extension PassThrough: Normalizer {
	public func adjust(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer)) {
	
	}
	public func flush(commandBuffer: MTLCommandBuffer) {
	
	}
}
