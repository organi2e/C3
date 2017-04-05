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
	let optimizer: (MTLCommandBuffer, MTLBuffer, MTLBuffer) -> Void
	public init(η: Float, count: Int) {
		optimizer = { (commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) in
			assert( θ.storageMode == .shared && count * MemoryLayout<Float>.size<=θ.length )
			assert( Δ.storageMode == .shared && count * MemoryLayout<Float>.size<=Δ.length )
			commandBuffer.addCompletedHandler { (_: MTLCommandBuffer) in
				cblas_saxpy(Int32(count), η, UnsafePointer<Float>(OpaquePointer(Δ.contents())), 1, UnsafeMutablePointer<Float>(OpaquePointer(θ.contents())), 1)
			}
		}
	}
	private init(pipeline: MTLComputePipelineState, count: Int) {
		
		optimizer = {
			
			let encoder: MTLComputeCommandEncoder = $0.0.makeComputeCommandEncoder()
			let threads: Int = pipeline.threadExecutionWidth

			assert( pipeline.device === encoder.device)
			assert( pipeline.device === $0.1.device && count * MemoryLayout<Float>.size <= $0.1.length )
			assert( pipeline.device === $0.2.device && count * MemoryLayout<Float>.size <= $0.2.length )
			
			encoder.setComputePipelineState(pipeline)
			encoder.setBuffer($0.1, offset: 0, at: 0)
			encoder.setBuffer($0.2, offset: 0, at: 1)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
	}
	public static func factory(η: Float = 1e-3) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let kernel: String = String(describing: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([η], type: .float, withName: "eta")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "\(kernel)Optimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				StochasticGradientDescent(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension StochasticGradientDescent: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		optimizer(commandBuffer, θ, Δ)
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		
	}
}
public typealias SGD = StochasticGradientDescent
