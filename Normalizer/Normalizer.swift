//
//  Normalizer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/06/12.
//
//
import Metal
public protocol Normalizer {
//	func statistics(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
//	func adjustment(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer))
	
	
	func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int)
	func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int)
	func connect(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: MTLBuffer, count: Int)
	func scaling(commandBuffer: MTLCommandBuffer, target: (μ: MTLBuffer, σ: MTLBuffer), source: (μ: MTLBuffer, σ: MTLBuffer), parameters: MTLBuffer, count: Int)
	func average(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
}
public class PassThrough {
	public init() {
		
	}
}
extension PassThrough: Normalizer {
	public func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int) {
		assert( commandBuffer.device === target.device && count * MemoryLayout<Float>.stride <= target.length )
		assert( commandBuffer.device === source.device && count * MemoryLayout<Float>.stride <= source.length )
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: count * MemoryLayout<Float>.stride)
		encoder.label = #function
		encoder.endEncoding()
	}
	public func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int) {
		assert( commandBuffer.device === target.device && count * MemoryLayout<Float>.stride <= target.length )
		assert( commandBuffer.device === source.device && count * MemoryLayout<Float>.stride <= source.length )
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: count * MemoryLayout<Float>.stride)
		encoder.label = #function
		encoder.endEncoding()
	}
	public func connect(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: MTLBuffer, count: Int) {
		
	}
	public func average(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
	
	}
	public func scaling(commandBuffer: MTLCommandBuffer, target: (μ: MTLBuffer, σ: MTLBuffer), source: (μ: MTLBuffer, σ: MTLBuffer), parameters: MTLBuffer, count: Int) {
	}
}
