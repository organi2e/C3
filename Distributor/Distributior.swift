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
	let D: MTLComputePipelineState
}
public protocol Corrector {
	func correct(j: (μ: MTLBuffer, σ: MTLBuffer), Δ: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func correct(χ: MTLBuffer, ϝ: MTLBuffer)
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
	let A: MTLComputePipelineState
	let B: MTLComputePipelineState
	let C: MTLComputePipelineState
	let D: MTLComputePipelineState
	let E: MTLComputePipelineState
	let F: MTLComputePipelineState
}
struct DeltaPipeline {
	let JP: MTLComputePipelineState
	let JV: MTLComputePipelineState
	let GP: MTLComputePipelineState
	let GV: MTLComputePipelineState
}
public protocol Derivator {
	func jacobian(commandBuffer: MTLCommandBuffer, x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(commandBuffer: MTLCommandBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer)
	func jacobian(commandBuffer: MTLCommandBuffer, b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(commandBuffer: MTLCommandBuffer, c: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(commandBuffer: MTLCommandBuffer, d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
	func jacobian(commandBuffer: MTLCommandBuffer, d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func flush(commandBuffer: MTLCommandBuffer)
	func fix(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
	func derivate(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer))
	func derivate(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer))
	func derivate(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer))
	var j: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) { get }
}
public protocol Distributor {
	func activate(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector)->Void)
	func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector)->Void)
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              x: MTLBuffer,
	              a: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              a: (μ: MTLBuffer, σ: MTLBuffer),
	              x: MTLBuffer, count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              b: (μ: MTLBuffer, σ: MTLBuffer),
	              y: MTLBuffer,
	              g: (μ: MTLBuffer, σ: MTLBuffer),
	              j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              c: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              d: MTLBuffer,
	              φ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              d: MTLBuffer,
	              φ: (μ: MTLBuffer, σ: MTLBuffer),
	              j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func jacobian(commandBuffer: MTLCommandBuffer,
	              j: (μ: MTLBuffer, σ: MTLBuffer),
	              Σ: (μ: MTLBuffer, σ: MTLBuffer),
	              φ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func derivate(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func derivate(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int))
	func flush(commandBuffer: MTLCommandBuffer, θ: MTLBuffer)
	func flush(commandBuffer: MTLCommandBuffer, θ: (μ: MTLBuffer, σ: MTLBuffer))
}
