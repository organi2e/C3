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
	func collect(d: MTLBuffer, Φ: (μ: MTLBuffer, σ: MTLBuffer))
}
struct CorrectorPipeline {
	let J: MTLComputePipelineState
	let G: MTLComputePipelineState
}
public protocol Corrector {
	func correct(j: (μ: MTLBuffer, σ: MTLBuffer), Δ: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func correct(Δ: MTLBuffer)
	var order: MTLCommandBuffer { get }
}
public protocol Activator {
	func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, collector: (Collector)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), corrector: (Corrector)->Void)
	var φ: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) { get }
	var g: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) { get }
}
struct JacobianPipeline {
	let X: MTLComputePipelineState
	let Y: MTLComputePipelineState
	let A: MTLComputePipelineState
	let B: MTLComputePipelineState
	let C: MTLComputePipelineState
	let D: MTLComputePipelineState
	let F: MTLComputePipelineState
}
public protocol Jacobian {
	func jacobian(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func jacobian(y: MTLBuffer, b: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int)
	func jacobian(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(c: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
}
public protocol Derivator {
	func derivate(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), jacobian: (Jacobian)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), jacobian: (Jacobian)->Void)
	var j: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) { get }
}
