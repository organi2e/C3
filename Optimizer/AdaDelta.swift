//
//  MomentumAdaDelta.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

import Metal
import simd
public class AdaDelta {
	let optimizer: MTLComputePipelineState
	let limit: Int
	let threads: MTLSize
	let groups: MTLSize
	let parameters: MTLBuffer
	private init(pipeline: MTLComputePipelineState, count: Int) {
		optimizer = pipeline
		limit = count
		threads = MTLSize(width: optimizer.threadExecutionWidth, height: 1, depth: 1)
		groups = MTLSize(width: (limit-1)/threads.width+1, height: 1, depth: 1)
		parameters = optimizer.device.makeBuffer(length: limit*MemoryLayout<float2>.size, options: .storageModePrivate)
	}
	public static func optimizer(device: MTLDevice, L2: Float = 0, L1: Float = 0, ρ: Float = 0.95, ε: Float = 1e-6) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([ρ], type: .float, withName: "rho")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let function: MTLFunction = try library.makeFunction(name: "AdaDeltaOptimize", constantValues: constantValues)
		let pipeline: MTLComputePipelineState = try device.makeComputePipelineState(function: function)
		return {
			AdaDelta(pipeline: pipeline, count: $0)
		}
	}
}
extension AdaDelta: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( optimizer.device === commandBuffer.device )
		assert( optimizer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( optimizer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.endEncoding()
	}
}

