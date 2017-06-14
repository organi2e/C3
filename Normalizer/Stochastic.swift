//
//  Normalizer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/06/09.
//
//

import Metal
import simd
public class Stochastic {
	let limit: Int
	let pipeline: MTLComputePipelineState
	let momentum: MTLBuffer
	let gradient: MTLBuffer
	private init(pipeline state: MTLComputePipelineState, count: Int) {
		let options: MTLResourceOptions = .storageModePrivate
		limit = count
		pipeline = state
		momentum = pipeline.device.makeBuffer(length: limit * MemoryLayout<float2>.stride, options: options)
		gradient = pipeline.device.makeBuffer(length: limit * MemoryLayout<float4>.stride, options: options)
	}
	public static func normalizer(device: MTLDevice, γ: Float = 0.99, ε: Float = 0) throws -> (Int) -> Normalizer {
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let pipeline: MTLComputePipelineState =
			try device.makeDefaultLibrary(bundle: Bundle(for: self)).make(name: "\(String(describing: self))Adjust", constantValues: constantValues)
		return {
			Stochastic(pipeline: pipeline, count: $0)
		}
	}
}
extension Stochastic: Normalizer {
	public func adjust(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer)) {
		
		assert( commandBuffer.device === pipeline.device )
		assert( commandBuffer.device === Δφ.μ.device && limit * MemoryLayout<Float>.stride <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && limit * MemoryLayout<Float>.stride <= Δφ.σ.length )
		assert( commandBuffer.device === momentum.device && limit * MemoryLayout<float2>.stride <= momentum.length )
		assert( commandBuffer.device === gradient.device && limit * MemoryLayout<float4>.stride <= gradient.length )
		assert( commandBuffer.device === φ.μ.device && limit * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && limit * MemoryLayout<Float>.stride <= φ.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = pipeline.threadExecutionWidth
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
		encoder.setBuffer(momentum, offset: 0, at: 2)
		encoder.setBuffer(gradient, offset: 0, at: 3)
		encoder.setBuffer(φ.μ, offset: 0, at: 4)
		encoder.setBuffer(φ.σ, offset: 0, at: 5)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 6)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
		
	}
	public func flush(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		[momentum, gradient].forEach {
			assert( $0.device === encoder.device )
			encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
		}
		encoder.label = #function
		encoder.endEncoding()
	}
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: makeFunction(name: name, constantValues: constantValues))
	}
}
