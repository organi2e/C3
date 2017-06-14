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
	let scaling: MTLComputePipelineState
	let average: MTLComputePipelineState
	public init(device: MTLDevice, γ: Float = 0.995, ε: Float = 0) throws {
		
		let Class: AnyClass = type(of: self)
		
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let bundle: Bundle = Bundle(for: Class)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		
		(collect, correct, connect, scaling, average)
			= try (library.make(name: "\(String(describing: Class))Collect", constantValues: constantValues),
			       library.make(name: "\(String(describing: Class))Correct", constantValues: constantValues),
			       library.make(name: "\(String(describing: Class))Connect", constantValues: constantValues),
			       library.make(name: "\(String(describing: Class))Scaling", constantValues: constantValues),
			       library.make(name: "\(String(describing: Class))Average", constantValues: constantValues))
	}
	/*
	public static func make(device: MTLDevice, γ: Float = 0.999, ε: Float = 0) throws {
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		
		let bundle: Bundle = Bundle(for: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		
		let (collect, correct, connect) = try (library.make(name: "\(String(describing: self))Collect", constantValues: constantValues),
		                                       library.make(name: "\(String(describing: self))Correct", constantValues: constantValues),
		                                       library.make(name: "\(String(describing: self))Connect", constantValues: constantValues)
		)
		
	}
	*/
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
	public func average(commandBuffer: MTLCommandBuffer, parameters: MTLBuffer, source: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( connect.device === commandBuffer.device )
		assert( connect.device === parameters.device && count * MemoryLayout<float4>.stride <= parameters.length )
		assert( connect.device === source.μ.device && count * MemoryLayout<Float>.stride <= source.μ.length )
		assert( connect.device === source.σ.device && count * MemoryLayout<Float>.stride <= source.σ.length )
		
		let threads: Int = average.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(average)
		encoder.setBuffer(parameters, offset: 0, at: 0)
		encoder.setBuffer(source.μ, offset: 0, at: 1)
		encoder.setBuffer(source.σ, offset: 0, at: 2)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func scaling(commandBuffer: MTLCommandBuffer, target: (μ: MTLBuffer, σ: MTLBuffer), source: (μ: MTLBuffer, σ: MTLBuffer), parameters: MTLBuffer, count: Int) {
		assert( connect.device === commandBuffer.device )
		assert( connect.device === parameters.device && count * MemoryLayout<float4>.stride <= parameters.length )
		assert( connect.device === target.μ.device && count * MemoryLayout<Float>.stride <= target.μ.length )
		assert( connect.device === target.σ.device && count * MemoryLayout<Float>.stride <= target.σ.length )
		assert( connect.device === source.μ.device && count * MemoryLayout<Float>.stride <= source.μ.length )
		assert( connect.device === source.σ.device && count * MemoryLayout<Float>.stride <= source.σ.length )
		
		let threads: Int = scaling.threadExecutionWidth
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(scaling)
		encoder.setBuffer(target.μ, offset: 0, at: 0)
		encoder.setBuffer(target.σ, offset: 0, at: 1)
		encoder.setBuffer(source.μ, offset: 0, at: 2)
		encoder.setBuffer(source.σ, offset: 0, at: 3)
		encoder.setBuffer(parameters, offset: 0, at: 4)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 5)
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
