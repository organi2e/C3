//
//  StochasticGradientDescent.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Accelerate
import Metal

public class StochasticGradientDescent {
	let optimizer: MTLComputePipelineState
	let limit: Int
	private init(pipeline: MTLComputePipelineState, count: Int) {
		optimizer = pipeline
		limit = count
	}
	public static func optimizer(device: MTLDevice, L2: Float = 0.0, L1: Float = 0.0, η: Float = 1e-3) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([L2, L1, η], type: .float3, withName: "eta")
		
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let function: MTLFunction = try library.makeFunction(name: "\(kernel)Optimize", constantValues: constantValues)
		let pipeline: MTLComputePipelineState = try device.makeComputePipelineState(function: function)
		return {
			StochasticGradientDescent(pipeline: pipeline, count: $0)
		}
	}
}
extension StochasticGradientDescent: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( optimizer.device === commandBuffer.device)
		assert( optimizer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( optimizer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = optimizer.threadExecutionWidth
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(Δ, offset: 0, at: 1)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		
	}
}
public typealias SGD = StochasticGradientDescent
