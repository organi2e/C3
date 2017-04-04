//
//  Momentum.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

import Metal

public class Momentum {
	let optimizer: MTLComputePipelineState
	let parameters: MTLBuffer
	let limit: Int
	private init(pipeline state: MTLComputePipelineState, count: Int) {
		limit = count
		optimizer = state
		parameters = state.device.makeBuffer(length: limit*MemoryLayout<Float>.size, options: .storageModePrivate)
	}
	public static func factory(η: Float = 1e-3, γ: Float = 0.9) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([η], type: .float, withName: "eta")
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "\(kernel)Optimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				Momentum(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Momentum: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {

		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = optimizer.threadExecutionWidth
		
		assert( optimizer.device === encoder.device )
		assert( optimizer.device === θ.device && parameters.length <= θ.length )
		assert( optimizer.device === Δ.device && parameters.length <= Δ.length )
		
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.endEncoding()
	}
}
