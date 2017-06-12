//
//  Normalizer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/06/12.
//
//
import Metal
public protocol Normalizer {
	func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer)
	func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer)
	func connect(commandBuffer: MTLCommandBuffer, source: MTLBuffer)
	func reset(commandBuffer: MTLCommandBuffer)
}
public class PassThrough {
	let limit: Int
	public init(count: Int) {
		limit = count
	}
}
extension PassThrough: Normalizer {
	public func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer) {
		assert( commandBuffer.device === target.device && limit * MemoryLayout<Float>.stride <= target.length )
		assert( commandBuffer.device === source.device && limit * MemoryLayout<Float>.stride <= source.length )
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: limit * MemoryLayout<Float>.stride)
		encoder.label = #function
		encoder.endEncoding()
	}
	public func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer) {
		assert( commandBuffer.device === target.device && limit * MemoryLayout<Float>.stride <= target.length )
		assert( commandBuffer.device === source.device && limit * MemoryLayout<Float>.stride <= source.length )
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: limit * MemoryLayout<Float>.stride)
		encoder.label = #function
		encoder.endEncoding()
	}
	public func connect(commandBuffer: MTLCommandBuffer, source: MTLBuffer) {
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
	
	}
}
