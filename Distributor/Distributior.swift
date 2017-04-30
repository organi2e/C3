//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
struct CollectPipeline {
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
struct CorrectPipeline {
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
struct ConnectPipeline {
	let X: MTLComputePipelineState
	let A: MTLComputePipelineState
	let B: MTLComputePipelineState
	let C: MTLComputePipelineState
	let D: MTLComputePipelineState
	let E: MTLComputePipelineState
	let F: MTLComputePipelineState
}
public protocol Connector {
	func connect(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer))
	func connect(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer)
	func connect(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer))
	func connect(c: (μ: MTLBuffer, σ: MTLBuffer))
	func connect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer))
	func connect(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer))
	var order: MTLCommandBuffer { get }
}
struct ActivatePipeline {
	let P: MTLComputePipelineState
	let V: MTLComputePipelineState
}
struct DerivatePipeline {
	let P: MTLComputePipelineState
	let V: MTLComputePipelineState
}
struct GradientPipeline {
	let JP: MTLComputePipelineState
	let JV: MTLComputePipelineState
	let GP: MTLComputePipelineState
	let GV: MTLComputePipelineState
}
public protocol Distributor {
	func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector)->Void)
	func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, correct: (Corrector)->Void)
	func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, correct: (Corrector)->Void)
	
	func gradient(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), connect: (Connector)->Void)
	func gradient(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), connect: (Connector)->Void)
	func gradient(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	              Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	              count: (rows: Int, cols: Int), connect: (Connector)->Void)
}
