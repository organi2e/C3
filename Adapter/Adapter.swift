//
//  Adapter.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public protocol Adapter {
	func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer)
	func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer)
}
public class Discard {
	public init(count: Int) {
		
	}
}
extension Discard: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
	}
}
public class Linear {
	let limit: Int
	public init(count: Int) {
		limit = count
	}
}
extension Linear: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.stride <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.stride <= φ.length )
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: φ, sourceOffset: 0, to: θ, destinationOffset: 0, size: limit * MemoryLayout<Float>.stride)
		encoder.label = "LinearAdapter.generate"
		encoder.endEncoding()
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		assert( commandBuffer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
	}
}
public class NonLinear {
	let generate: MTLComputePipelineState
	let gradient: MTLComputePipelineState
	let limit: Int
	internal init(pipeline: (MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		generate = pipeline.0
		gradient = pipeline.1
		limit = count
	}
	internal static func compile(device: MTLDevice) throws -> (MTLComputePipelineState, MTLComputePipelineState) {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		return (try library.make(name: "\(kernel)Generate"), try library.make(name: "\(kernel)Gradient"))
	}
}
extension NonLinear: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === generate.device )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = generate.threadExecutionWidth
		encoder.setComputePipelineState(generate)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(φ, offset: 0, at: 1)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = "\(String(describing: self)).generate(\(limit))"
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === gradient.device )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = gradient.threadExecutionWidth
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = "\(String(describing: self)).gradient(\(limit))"
		encoder.endEncoding()
		
	}
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues())
		throws -> MTLComputePipelineState {
			return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
