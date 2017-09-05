//
//  Optimizer.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//  ref: http://qiita.com/skitaoka/items/e6afbe238cd69c899b2a
//

import Metal

public protocol Optimizer {
	func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer)
	func reset(commandBuffer: MTLCommandBuffer)
}
