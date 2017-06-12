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
	let collect: MTLComputePipelineState
	let correct: MTLComputePipelineState
	let connect: MTLComputePipelineState
	public init(device: MTLDevice, γ: Float = 0.99, ε: Float = 0) throws {
		
		let `Self`: AnyClass = type(of: self)
		
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let bundle: Bundle = Bundle(for: Self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		
		(collect, correct, connect) = try (library.make(name: "\(String(describing: Self))Collect", constantValues: constantValues),
		                                   library.make(name: "\(String(describing: Self))Correct", constantValues: constantValues),
		                                   library.make(name: "\(String(describing: Self))Connect", constantValues: constantValues)
		)
		
	}
}
extension Stochastic: Normalizer {
	public func collect(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int) {
		
		assert( collect.device === commandBuffer.device )
		assert( collect.device === parameters.device && count * MemoryLayout<float2>.stride <= parameters.length )
		assert( collect.device === target.device && count * MemoryLayout<Float>.stride <= target.length )
		assert( collect.device === source.device && count * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = collect.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(collect)
		encoder.setBuffer(target, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(source, offset: 0, at: 2)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func correct(commandBuffer: MTLCommandBuffer, target: MTLBuffer, source: MTLBuffer, parameters: MTLBuffer, count: Int) {
		
		assert( correct.device === commandBuffer.device )
		assert( correct.device === parameters.device && count * MemoryLayout<float2>.stride <= parameters.length )
		assert( correct.device === target.device && count * MemoryLayout<Float>.stride <= target.length )
		assert( correct.device === source.device && count * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = correct.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(correct)
		encoder.setBuffer(target, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(source, offset: 0, at: 2)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func connect(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: MTLBuffer, count: Int) {
		
		assert( connect.device === commandBuffer.device )
		assert( connect.device === parameters.device && count * MemoryLayout<float2>.stride <= parameters.length )
		assert( connect.device === source.device && count * MemoryLayout<Float>.stride <= source.length )
		
		let threads: Int = connect.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(connect)
		encoder.setBuffer(parameters, offset: 0, at: 0)
		encoder.setBuffer(source, offset: 0, at: 1)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: makeFunction(name: name, constantValues: constantValues))
	}
}
