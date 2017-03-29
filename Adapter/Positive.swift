//
//  Positive.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Metal
public class Positive {
	let generate: MTLComputePipelineState
	let gradient: MTLComputePipelineState
	let limit: Int
	public var L1: Float
	public var L2: Float
	private init(pipeline: (MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		generate = pipeline.0
		gradient = pipeline.1
		limit = count
		L1 = 0
		L2 = 0
	}
	public static func factory() -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let generate: MTLComputePipelineState = try library.make(name: "PositiveGenerate")
			let gradient: MTLComputePipelineState = try library.make(name: "PositiveGradient")
			return {
				Positive(pipeline: (generate, gradient), count: $0)
			}
		}
	}
}
extension Positive: Adapter {
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
		encoder.setBytes(&L1, length: MemoryLayout.size(ofValue: L1), at: 3)
		encoder.setBytes(&L2, length: MemoryLayout.size(ofValue: L2), at: 4)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
}
