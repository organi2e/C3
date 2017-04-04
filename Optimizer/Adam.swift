//
//  Adam.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/02/01.
//
//

import Metal
import simd

public class Adam {
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
		parameters = pipeline.device.makeBuffer(length: limit * MemoryLayout<float2>.size, options: .storageModePrivate)
	}
	public static func factory(α: Float = 1e-3, β: Float = 0.9, γ: Float = 0.999, ε: Float = 0) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([α], type: .float, withName: "alpha")
		constantValues.setConstantValue([β], type: .float, withName: "beta")
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "\(kernel)Optimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				Adam(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Adam: Optimizer {
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

