//
//  Adapter.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public protocol Adapter {
	func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer)
	func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer)
	//func adapt(commandBuffer: MTLCommandBuffer, φ: MTLBuffer, θ: MTLBuffer, Δ: MTLBuffer)
}
internal extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
