//
//  Regular.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public class Regular {
	let generate: MTLComputePipelineState
	let gradient: MTLComputePipelineState
	let adapt: MTLComputePipelineState
	let limit: Int
	private init(pipeline: (MTLComputePipelineState, MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		generate = pipeline.0
		gradient = pipeline.1
		adapt = pipeline.2
		limit = count
	}
	public static func factory() -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let generate: MTLComputePipelineState = try library.make(name: "RegularGenerate")
			let gradient: MTLComputePipelineState = try library.make(name: "RegularGradient")
			let adapt: MTLComputePipelineState = try library.make(name: "RegularAdapt")
			return {
				Regular(pipeline: (generate, gradient, adapt), count: $0)
			}
		}
	}
}
extension Regular: Adapter {
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
		
		assert( gradient.device === commandBuffer.device )
		assert( gradient.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		assert( gradient.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( gradient.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
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
	public func adapt(commandBuffer: MTLCommandBuffer, φ: MTLBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( adapt.device === commandBuffer.device )
		assert( adapt.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		assert( adapt.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( adapt.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = adapt.threadExecutionWidth
		encoder.setComputePipelineState(adapt)
		encoder.setBuffer(φ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
}
