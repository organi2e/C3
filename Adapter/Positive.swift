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
	private init(pipeline: (MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		generate = pipeline.0
		gradient = pipeline.1
		limit = count
	}
	public static func adapter(L1: Float = 0, L2: Float = 0) -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([1, L1, L2], type: .float3, withName: "LRW")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let generate: MTLComputePipelineState = try library.make(name: "\(kernel)Generate")
			let gradient: MTLComputePipelineState = try library.make(name: "\(kernel)Gradient", constantValues: constantValues)
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
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
}
