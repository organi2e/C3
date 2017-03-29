//
//  Optimizer.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Metal

public protocol Optimizer {
	func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer)
	func reset(commandBuffer: MTLCommandBuffer)
}
