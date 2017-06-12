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
	let parameters: MTLBuffer
	let collect: MTLComputePipelineState
	let correct: MTLComputePipelineState
	let connect: MTLComputePipelineState
	private init(device: MTLDevice,
	             pipeline: (collect: MTLComputePipelineState, correct: MTLComputePipelineState, connect: MTLComputePipelineState),
	             count: Int) {
		
		collect = pipeline.collect
		correct = pipeline.correct
		connect = pipeline.connect
		
		limit = count
		parameters = device.makeBuffer(bytes: Array<float2>(repeating: float2(0, 1), count: limit),
		                               length: limit * MemoryLayout<float2>.stride, options: .storageModePrivate)
//		parameters = device.makeBuffer(length: count * MemoryLayout<float2>.stride, options: .storageModePrivate)
		parameters.label = #function
	}
	public static func make(device: MTLDevice, γ: Float = 0.995, ε: Float = 0) throws -> (Int) -> Normalizer {
		
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let bundle: Bundle = Bundle(for: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let pipeline: (collect: MTLComputePipelineState, correct: MTLComputePipelineState, connect: MTLComputePipelineState)
			= try(collect: library.make(name: "\(String(describing: self))Collect", constantValues: constantValues),
			      correct: library.make(name: "\(String(describing: self))Correct", constantValues: constantValues),
			      connect: library.make(name: "\(String(describing: self))Connect", constantValues: constantValues))
		return {
			Stochastic(device: device, pipeline: pipeline, count: $0)
		}
	}
}
extension Stochastic: Normalizer {
	public func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer) {
		
		assert( collect.device === commandBuffer.device )
		assert( collect.device === parameters.device && limit * MemoryLayout<float2>.stride <= parameters.length )
		assert( collect.device === target.device && limit * MemoryLayout<Float>.stride <= target.length )
		assert( collect.device === source.device && limit * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = collect.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(collect)
		encoder.setBuffer(target, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(source, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer) {
		
		assert( correct.device === commandBuffer.device )
		assert( correct.device === parameters.device && limit * MemoryLayout<float2>.stride <= parameters.length )
		assert( correct.device === target.device && limit * MemoryLayout<Float>.stride <= target.length )
		assert( correct.device === source.device && limit * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = correct.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(correct)
		encoder.setBuffer(target, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(source, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func connect(commandBuffer: MTLCommandBuffer, source: MTLBuffer) {
		
		assert( connect.device === commandBuffer.device )
		assert( connect.device === parameters.device && limit * MemoryLayout<float2>.stride <= parameters.length )
		assert( connect.device === source.device && limit * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = connect.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(connect)
		encoder.setBuffer(parameters, offset: 0, at: 0)
		encoder.setBuffer(source, offset: 0, at: 1)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (limit-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.label = #function
		encoder.endEncoding()
	}
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: makeFunction(name: name, constantValues: constantValues))
	}
}
