//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import Metal
import simd

public class GaussDistributor {
	let collectPipeline: CollectPipeline
	let correctPipeline: CorrectPipeline
	let connectPipeline: ConnectPipeline
	let activatePipeline: ActivatePipeline
	let derivatePipeline: DerivatePipeline
	let gradientPipeline: GradientPipeline
	public init(device: MTLDevice, xorshift: (Int, Int, Int) = (5, 7, 4)) throws {
		let bundle: Bundle = Bundle(for: GaussDistributor.self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let xorshiftValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		xorshiftValues.setConstantValue([uint3(UInt32(xorshift.0), UInt32(xorshift.1), UInt32(xorshift.2))], type: .uint3, withName: "xorshift16")
		collectPipeline = try CollectPipeline(
			W: library.make(name: "GaussCollectW"),
			C: library.make(name: "GaussCollectC"),
			D: library.make(name: "GaussCollectD"),
			F: library.make(name: "GaussCollectF")
		)
		correctPipeline = try CorrectPipeline(
			J: library.make(name: "GaussCorrectJ"),
			G: library.make(name: "GaussCorrectG"),
			N: library.make(name: "GaussCorrectN"),
			P: library.make(name: "GaussCorrectP"),
			V: library.make(name: "GaussCorrectV")
		)
		connectPipeline = try ConnectPipeline(
			X: library.make(name: "GaussConnectX"),
			A: library.make(name: "GaussConnectA"),
			B: library.make(name: "GaussConnectB"),
			C: library.make(name: "GaussConnectC"),
			D: library.make(name: "GaussConnectD"),
			E: library.make(name: "GaussConnectE"),
			F: library.make(name: "GaussConnectF")
		)
		activatePipeline = try ActivatePipeline(
			P: library.make(name: "GaussActivateP", constantValues: xorshiftValues),
			V: library.make(name: "GaussActivateV", constantValues: xorshiftValues),
			X: library.make(name: "GaussActivateX")
		)
		derivatePipeline = try DerivatePipeline(
			P: library.make(name: "GaussDerivateP"),
			V: library.make(name: "GaussDerivateV"),
			N: library.make(name: "GaussDerivateN")
		)
		gradientPipeline = try GradientPipeline(
			JP: library.make(name: "GaussGradientJP"),
			JV: library.make(name: "GaussGradientJV"),
			GP: library.make(name: "GaussGradientGP"),
			GV: library.make(name: "GaussGradientGV")
		)
	}
}
extension GaussDistributor {
	private struct GaussCollector: Collector {
		let order: MTLCommandBuffer
		let state: CollectPipeline
		let width: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		public func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			
			assert( order.device === state.W.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === w.μ.device && width * count * MemoryLayout<Float>.stride <= w.μ.length )
			assert( order.device === w.σ.device && width * count * MemoryLayout<Float>.stride <= w.σ.length )
			assert( order.device === x.device && count * MemoryLayout<Float>.stride <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.W.threadExecutionWidth
			encoder.setComputePipelineState(state.W)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(w.μ, offset: 0, at: 2)
			encoder.setBuffer(w.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(width), y: uint(count))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.setThreadgroupMemoryLength(threads*MemoryLayout<float2x4>.stride, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
		public func collect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.stride <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.stride <= c.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(c.μ, offset: 0, at: 2)
			encoder.setBuffer(c.σ, offset: 0, at: 3)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
		public func collect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(d, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector) -> Void) {
		do {
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: φ.μ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.fill(buffer: φ.σ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.label = #function
			encoder.endEncoding()
		}
		do {
			collect(GaussCollector(order: commandBuffer, state: collectPipeline, width: count, Σ: φ))
		}
		do {
			assert( commandBuffer.device === collectPipeline.F.device )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = collectPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(collectPipeline.F)
			encoder.setBuffer(φ.μ, offset: 0, at: 0)
			encoder.setBuffer(φ.σ, offset: 0, at: 1)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( commandBuffer.device === activatePipeline.P.device )
		assert( commandBuffer.device === f.device && count * MemoryLayout<Float>.stride <= f.length )
		assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		typealias T = ushort
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = activatePipeline.P.threadExecutionWidth
		let uniform: Array<T> = Array<Void>(repeating: (), count: 4 * threads).map {
			T(1) + T(arc4random_uniform(UInt32(T.max-1)))
		}
		encoder.setComputePipelineState(activatePipeline.P)
		encoder.setBuffer(f, offset: 0, at: 0)
		encoder.setBuffer(g.μ, offset: 0, at: 1)
		encoder.setBuffer(g.σ, offset: 0, at: 2)
		encoder.setBuffer(φ.μ, offset: 0, at: 3)
		encoder.setBuffer(φ.σ, offset: 0, at: 4)
		encoder.setBytes(uniform, length: uniform.count * MemoryLayout<T>.stride, at: 5)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
		encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( commandBuffer.device === activatePipeline.V.device )
		assert( commandBuffer.device === v.device && count * MemoryLayout<Float>.stride <= v.length )
		assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		typealias T = ushort
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = activatePipeline.V.threadExecutionWidth
		let uniform: Array<T> = Array<Void>(repeating: (), count: 4 * threads).map { T(1) + T(arc4random_uniform(UInt32(T.max-1))) }
		encoder.setComputePipelineState(activatePipeline.V)
		encoder.setBuffer(v, offset: 0, at: 0)
		encoder.setBuffer(g.μ, offset: 0, at: 1)
		encoder.setBuffer(g.σ, offset: 0, at: 2)
		encoder.setBuffer(φ.μ, offset: 0, at: 3)
		encoder.setBuffer(φ.σ, offset: 0, at: 4)
		encoder.setBytes(uniform, length: uniform.count * MemoryLayout<T>.stride, at: 5)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
		encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector) -> Void) {
		do {
			activate(commandBuffer: commandBuffer, φ: φ, count: count, collect: collect)
		}
		do {
			assert( commandBuffer.device === activatePipeline.P.device )
			assert( commandBuffer.device === f.device && count * MemoryLayout<Float>.stride <= f.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			typealias T = ushort
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatePipeline.P.threadExecutionWidth
			let uniform: Array<T> = Array<Void>(repeating: (), count: 4 * threads).map {
				T(1) + T(arc4random_uniform(UInt32(T.max-1)))
			}
			encoder.setComputePipelineState(activatePipeline.P)
			encoder.setBuffer(f, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(g.σ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			encoder.setBytes(uniform, length: uniform.count * MemoryLayout<T>.stride, at: 5)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector) -> Void) {
		do {
			activate(commandBuffer: commandBuffer, φ: φ, count: count, collect: collect)
		}
		do {
			assert( commandBuffer.device === activatePipeline.V.device )
			assert( commandBuffer.device === v.device && count * MemoryLayout<Float>.stride <= v.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			typealias T = ushort
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatePipeline.V.threadExecutionWidth
			let uniform: Array<T> = Array<Void>(repeating: (), count: 4 * threads).map { T(1) + T(arc4random_uniform(UInt32(T.max-1))) }
			encoder.setComputePipelineState(activatePipeline.V)
			encoder.setBuffer(v, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(g.σ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			encoder.setBytes(uniform, length: uniform.count * MemoryLayout<T>.stride, at: 5)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, p: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( commandBuffer.device === activatePipeline.X.device )
		assert( commandBuffer.device === p.device && count * MemoryLayout<Float>.stride <= p.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = activatePipeline.X.threadExecutionWidth
		encoder.setComputePipelineState(activatePipeline.X)
		encoder.setBuffer(p, offset: 0, at: 0)
		encoder.setBuffer(φ.μ, offset: 0, at: 1)
		encoder.setBuffer(φ.σ, offset: 0, at: 2)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
}
extension GaussDistributor {
	private struct GaussCorrector: Corrector {
		let order: MTLCommandBuffer
		let state: CorrectPipeline
		let width: Int
		let Δ: MTLBuffer
		public func correct(χ: MTLBuffer, ϝ: MTLBuffer) {
			assert( order.device === state.G.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length)
			assert( order.device === χ.device && width * MemoryLayout<Float>.stride <= χ.length)
			assert( order.device === ϝ.device && width * MemoryLayout<Float>.stride <= ϝ.length)
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.G)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(χ, offset: 0, at: 1)
			encoder.setBuffer(ϝ, offset: 0, at: 2)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
		public func correct(χ: MTLBuffer) {
			assert( order.device === state.N.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length)
			assert( order.device === χ.device && width * MemoryLayout<Float>.stride <= χ.length)
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.N.threadExecutionWidth
			encoder.setComputePipelineState(state.N)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(χ, offset: 0, at: 1)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
		public func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer) {
			assert( order.device === state.G.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			assert( order.device === f.device && width * MemoryLayout<Float>.stride <= f.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.P)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(φ.σ, offset: 0, at: 2)
			encoder.setBuffer(f, offset: 0, at: 3)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
		public func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer) {
			assert( order.device === state.V.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			assert( order.device === v.device && width * MemoryLayout<Float>.stride <= v.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.V)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(φ.σ, offset: 0, at: 2)
			encoder.setBuffer(v, offset: 0, at: 3)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, count: Int, correct: (Corrector) -> Void) {
		do {
			assert( commandBuffer.device === Δ.device && count * MemoryLayout<Float>.stride <= Δ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.label = #function
			encoder.endEncoding()
		}
		do {
			correct(GaussCorrector(order: commandBuffer, state: correctPipeline, width: count, Δ: Δ))
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), Δ: MTLBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( commandBuffer.device === derivatePipeline.P.device )
		assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.stride <= Δφ.σ.length )
		assert( commandBuffer.device === Δ.device && count * MemoryLayout<Float>.stride <= Δ.length )
		assert( commandBuffer.device === f.device && count * MemoryLayout<Float>.stride <= f.length )
		assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = derivatePipeline.P.threadExecutionWidth
		encoder.setComputePipelineState(derivatePipeline.P)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBuffer(f, offset: 0, at: 3)
		encoder.setBuffer(g.μ, offset: 0, at: 4)
		encoder.setBuffer(g.σ, offset: 0, at: 5)
		encoder.setBuffer(φ.μ, offset: 0, at: 6)
		encoder.setBuffer(φ.σ, offset: 0, at: 7)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 8)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), Δ: MTLBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		assert( commandBuffer.device === activatePipeline.V.device )
		assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.stride <= Δφ.σ.length )
		assert( commandBuffer.device === Δ.device && count * MemoryLayout<Float>.stride <= Δ.length )
		assert( commandBuffer.device === v.device && count * MemoryLayout<Float>.stride <= v.length )
		assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.stride <= g.σ.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = derivatePipeline.V.threadExecutionWidth
		encoder.setComputePipelineState(derivatePipeline.V)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBuffer(v, offset: 0, at: 3)
		encoder.setBuffer(g.μ, offset: 0, at: 4)
		encoder.setBuffer(g.σ, offset: 0, at: 5)
		encoder.setBuffer(φ.μ, offset: 0, at: 6)
		encoder.setBuffer(φ.σ, offset: 0, at: 7)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 8)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), θ: (μ: MTLBuffer, g: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), γ: Float, count: Int) {
		assert( commandBuffer.device === derivatePipeline.N.device )
		assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.stride <= Δφ.σ.length )
		assert( commandBuffer.device === θ.μ.device && count * MemoryLayout<Float>.stride <= θ.μ.length )
		assert( commandBuffer.device === θ.g.device && count * MemoryLayout<Float>.stride <= θ.g.length )
		assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = derivatePipeline.N.threadExecutionWidth
		encoder.setComputePipelineState(derivatePipeline.N)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
		encoder.setBuffer(θ.μ, offset: 0, at: 2)
		encoder.setBuffer(θ.g, offset: 0, at: 3)
		encoder.setBuffer(φ.μ, offset: 0, at: 4)
		encoder.setBuffer(φ.σ, offset: 0, at: 5)
		encoder.setBytes([γ], length: MemoryLayout<Float>.size, at: 6)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 7)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.label = #function
		encoder.endEncoding()
	}
}
extension GaussDistributor {
	private struct GaussConnector: Connector {
		let order: MTLCommandBuffer
		let state: ConnectPipeline
		let width: Int
		let refer: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		func connect(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.X.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.stride <= x.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.stride <= a.μ.length )
			assert( order.device === a.σ.device && width * refer * MemoryLayout<Float>.stride <= a.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.X.threadExecutionWidth
			encoder.setComputePipelineState(state.X)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBuffer(a.μ, offset: 0, at: 3)
			encoder.setBuffer(a.σ, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(width), y: uint(refer))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
			
		}
		func connect(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer) {
			
			assert( order.device === state.A.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.stride <= a.μ.length )
			assert( order.device === a.σ.device && width * refer * MemoryLayout<Float>.stride <= a.σ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.stride <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.A.threadExecutionWidth
			encoder.setComputePipelineState(state.A)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(a.μ, offset: 0, at: 2)
			encoder.setBuffer(a.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(width), y: uint(refer))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
			
		}
		func connect(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
		}
		func connect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.stride <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.stride <= c.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(c.μ, offset: 0, at: 2)
			encoder.setBuffer(c.σ, offset: 0, at: 3)
			encoder.setBytes([uint2(x: uint(width), y: uint(refer))], length: MemoryLayout<uint2>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
			
		}
		func connect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(d, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(width), y: uint(refer))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
			
		}
		func connect(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.E.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.stride <= Σ.σ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === j.μ.device && width * refer * MemoryLayout<Float>.stride <= j.μ.length )
			assert( order.device === j.σ.device && width * refer * MemoryLayout<Float>.stride <= j.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.E.threadExecutionWidth
			encoder.setComputePipelineState(state.E)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.σ, offset: 0, at: 3)
			encoder.setBuffer(d, offset: 0, at: 4)
			encoder.setBuffer(j.μ, offset: 0, at: 5)
			encoder.setBuffer(j.σ, offset: 0, at: 6)
			encoder.setBytes([uint2(x: uint(width), y: uint(refer))], length: MemoryLayout<uint2>.size, at: 7)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
			
		}
	}
	private func gradient(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		do {
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: count.rows * count.cols * MemoryLayout<Float>.stride), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: count.rows * count.cols * MemoryLayout<Float>.stride), value: 0)
			encoder.label = #function
			encoder.endEncoding()
		}
		do {
			connect(GaussConnector(order: commandBuffer, state: connectPipeline, width: count.rows, refer: count.cols, Σ: j))
		}
		do {
			assert( commandBuffer.device === connectPipeline.F.device )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			assert( commandBuffer.device === φ.μ.device && count.rows * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count.rows * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = connectPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(connectPipeline.F)
			encoder.setBuffer(j.μ, offset: 0, at: 0)
			encoder.setBuffer(j.σ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.σ, offset: 0, at: 3)
			encoder.setBytes([uint2(x: uint(count.rows), y: uint(count.cols))], length: MemoryLayout<uint2>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth : 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		gradient(commandBuffer: commandBuffer, j: j, φ: φ, count: count, connect: connect)
		do {
			assert( commandBuffer.device === gradientPipeline.JV.device )
			assert( commandBuffer.device === Δx.device && count.cols * MemoryLayout<Float>.stride <= Δx.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.JV.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.JV)
			encoder.setBuffer(Δx, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(count.cols), y: uint(count.rows))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.setThreadgroupMemoryLength(threads*MemoryLayout<float4>.stride, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (count.cols+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		gradient(commandBuffer: commandBuffer, j: j, φ: φ, count: count, connect: connect)
		do {
			assert( commandBuffer.device === gradientPipeline.GV.device )
			assert( commandBuffer.device === Δv.device && count.rows * MemoryLayout<Float>.stride <= Δv.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.GV)
			encoder.setBuffer(Δv, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
			encoder.setBytes([uint2(x: uint(count.rows), y: uint(count.cols))], length: MemoryLayout<uint2>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		gradient(commandBuffer: commandBuffer, j: j, φ: φ, count: count, connect: connect)
		do {
			assert( commandBuffer.device === gradientPipeline.GP.device )
			assert( commandBuffer.device === Δθ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δθ.μ.length )
			assert( commandBuffer.device === Δθ.σ.device && count.rows * MemoryLayout<Float>.stride <= Δθ.σ.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.GP)
			encoder.setBuffer(Δθ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δθ.σ, offset: 0, at: 1)
			encoder.setBuffer(j.μ, offset: 0, at: 2)
			encoder.setBuffer(j.σ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 4)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 5)
			encoder.setBytes([uint2(x: uint(count.rows), y: uint(count.cols))], length: MemoryLayout<uint2>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = #function
			encoder.endEncoding()
		}
	}
}
extension GaussDistributor: Distributor {

}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: makeFunction(name: name, constantValues: constantValues))
	}
}
