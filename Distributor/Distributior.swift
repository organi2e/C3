//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
struct CollectorPipeline {
	let W: MTLComputePipelineState
	let C: MTLComputePipelineState
	let D: MTLComputePipelineState
	let F: MTLComputePipelineState
}
public protocol Collector {
	func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int)
	func collect(c: (μ: MTLBuffer, σ: MTLBuffer))
	func collect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
	var order: MTLCommandBuffer { get }
}
struct CorrectorPipeline {
	let J: MTLComputePipelineState
	let G: MTLComputePipelineState
	let N: MTLComputePipelineState
	let P: MTLComputePipelineState
	let V: MTLComputePipelineState
}
public protocol Corrector {
	func correct(χ: MTLBuffer, ϝ: MTLBuffer)
	func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer)
	func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer)
	var Δ: MTLBuffer { get }
	var order: MTLCommandBuffer { get }
}
public protocol Jacobian {
	func jacobian(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer)
	func jacobian(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(c: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer))
	var order: MTLCommandBuffer { get }
}
struct JacobianPipeline {
	let X: MTLComputePipelineState
	let A: MTLComputePipelineState
	let B: MTLComputePipelineState
	let C: MTLComputePipelineState
	let D: MTLComputePipelineState
	let E: MTLComputePipelineState
	let F: MTLComputePipelineState
}
struct ActivatorPipeline {
	let AP: MTLComputePipelineState
	let AV: MTLComputePipelineState
	let GP: MTLComputePipelineState
	let GV: MTLComputePipelineState
}
struct DerivatorPipeline {
	let JP: MTLComputePipelineState
	let JV: MTLComputePipelineState
	let GP: MTLComputePipelineState
	let GV: MTLComputePipelineState
}
public protocol Distributor {
	func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector)->Void)
	func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector)->Void)
	func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector)->Void)
	func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector)->Void)
	
	func derivate(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void)
}
