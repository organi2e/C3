//
//  SMORMS3.swift
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

import Metal
import simd
public class SMORMS3 {
	let optimizer: MTLComputePipelineState
	let parameters: MTLBuffer
	let limit: Int
	private init(pipeline: MTLComputePipelineState, count: Int) {
		limit = count
		optimizer = pipeline
		parameters = pipeline.device.makeBuffer(length: count*MemoryLayout<float3>.size, options: .storageModePrivate)
	}
	public static func factory(α: Float = 0.9, ε: Float = 0.0) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([α], type: .float, withName: "alpha")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "SMORMS3Optimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				SMORMS3(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension SMORMS3: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( optimizer.device === commandBuffer.device )
		assert( optimizer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( optimizer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = optimizer.threadExecutionWidth
		
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
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
