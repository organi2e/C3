//
//  SMORMS3.swift
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

import Metal

public class SMORMS3 {
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
		parameters = optimizer.device.makeBuffer(length: 4*((3*limit-1)/4+1) * MemoryLayout<Float>.stride,
		                                         options: .storageModePrivate)
		parameters.label = "SMORMS3.Parameters(\(limit))"
	}
	public static func optimizer(device: MTLDevice, L2: Float = 0.0, L1: Float = 0.0, α: Float = 1e-3, ε: Float = 0.0) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([L2, L1, α], type: .float3, withName: "alpha")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let function: MTLFunction = try library.makeFunction(name: "\(kernel)Optimize", constantValues: constantValues)
		let pipeline: MTLComputePipelineState = try device.makeComputePipelineState(function: function)
		return {
			SMORMS3(pipeline: pipeline, count: $0)
		}
	}
}
extension SMORMS3: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( optimizer.device === commandBuffer.device )
		assert( optimizer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( optimizer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		assert( optimizer.device === parameters.device && 3 * limit * MemoryLayout<Float>.size <= parameters.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBufferOffset(0, at: 1)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.label = "SMORMS3.Optimize(\(limit))"
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: 3 * limit * MemoryLayout<Float>.size), value: 0)
		encoder.label = "SMORMS3.Reset(\(limit))"
		encoder.endEncoding()
	}
}
