//
//  Degenerate.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/10.
//
//

import Accelerate
import Metal
import simd

public class DegenerateDistributor {
	let collectPipeline: CollectPipeline
	let correctPipeline: CorrectPipeline
	let connectPipeline: ConnectPipeline
	let activatePipeline: ActivatePipeline
	let derivatePipeline: DerivatePipeline
	let gradientPipeline: GradientPipeline
	public init(device: MTLDevice) throws {
		let bundle: Bundle = Bundle(for: DegenerateDistributor.self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		collectPipeline = CollectPipeline(
			W: try library.make(name: "DegenerateCollectW"),
			C: try library.make(name: "DegenerateCollectC"),
			D: try library.make(name: "DegenerateCollectD"),
			F: try library.make(name: "DegenerateCollectF")
		)
		correctPipeline = CorrectPipeline(
			J: try library.make(name: "DegenerateCorrectJ"),
			G: try library.make(name: "DegenerateCorrectG"),
			N: try library.make(name: "DegenerateCorrectN"),
			P: try library.make(name: "DegenerateCorrectP"),
			V: try library.make(name: "DegenerateCorrectV")
		)
		connectPipeline = ConnectPipeline(
			X: try library.make(name: "DegenerateConnectX"),
			A: try library.make(name: "DegenerateConnectA"),
			B: try library.make(name: "DegenerateConnectB"),
			C: try library.make(name: "DegenerateConnectC"),
			D: try library.make(name: "DegenerateConnectD"),
			E: try library.make(name: "DegenerateConnectE"),
			F: try library.make(name: "DegenerateConnectF")
		)
		activatePipeline = ActivatePipeline(
			P: try library.make(name: "DegenerateActivateP"),
			V: try library.make(name: "DegenerateActivateV")
		)
		derivatePipeline = DerivatePipeline(
			P: try library.make(name: "DegenerateDerivateP"),
			V: try library.make(name: "DegenerateDerivateV")
		)
		gradientPipeline = GradientPipeline(
			JP: try library.make(name: "DegenerateGradientJ"),
			JV: try library.make(name: "DegenerateGradientJ"),
			GP: try library.make(name: "DegenerateGradientG"),
			GV: try library.make(name: "DegenerateGradientG")
		)
	}

}
extension DegenerateDistributor {
	private struct DegenerateCollector: Collector {
		let order: MTLCommandBuffer
		let state: CollectPipeline
		let width: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		public func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			
			assert( order.device === state.W.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === w.μ.device && width * count * MemoryLayout<Float>.stride <= w.μ.length )
			assert( order.device === x.device && count * MemoryLayout<Float>.stride <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.W.threadExecutionWidth
			encoder.setComputePipelineState(state.W)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(w.μ, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(count))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.setThreadgroupMemoryLength(threads*MemoryLayout<float4>.stride, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.CollectW(\(width, count))"
			encoder.endEncoding()
		}
		public func collect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.stride <= c.μ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(c.μ, offset: 0, at: 1)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.CollectC(\(width))"
			encoder.endEncoding()
		}
		public func collect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(d, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.CollectD(\(width))"
			encoder.endEncoding()
		}
	}
	private func activate(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector) -> Void) {
		do {
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: φ.μ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.label = "Degenerate.CollectFlush(\(count))"
			encoder.endEncoding()
		}
		collect(DegenerateCollector(order: commandBuffer, state: collectPipeline, width: count, Σ: φ))
	}
	public func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collect: (Collector) -> Void) {
		do {
			activate(commandBuffer: commandBuffer, φ: φ, count: count, collect: collect)
		}
		do {
			assert( commandBuffer.device === activatePipeline.P.device )
			assert( commandBuffer.device === f.device && count * MemoryLayout<Float>.stride <= f.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatePipeline.P.threadExecutionWidth
			encoder.setComputePipelineState(activatePipeline.P)
			encoder.setBuffer(f, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ActivateP(\(count))"
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
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatePipeline.V.threadExecutionWidth
			encoder.setComputePipelineState(activatePipeline.V)
			encoder.setBuffer(v, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ActivateV(\(count))"
			encoder.endEncoding()
		}
	}
}
extension DegenerateDistributor {
	private struct DegenerateCorrector: Corrector {
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
			encoder.endEncoding()
		}
		public func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer) {
			assert( order.device === state.P.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === f.device && width * MemoryLayout<Float>.stride <= f.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.P)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(f, offset: 0, at: 2)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.CorrectP(\(width))"
			encoder.endEncoding()
		}
		public func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer) {
			assert( order.device === state.V.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === v.device && width * MemoryLayout<Float>.stride <= v.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.V)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(v, offset: 0, at: 2)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.CorrectV(\(width))"
			encoder.endEncoding()
		}
	}
	private func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, correct: (Corrector)->Void) {
		do {
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.stride <= Δφ.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δφ.μ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.label = "Degenerate.CorrectFlush(\(count))"
			encoder.endEncoding()
		}
		correct(DegenerateCorrector(order: commandBuffer, state: correctPipeline, width: count, Δ: Δφ.μ))
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, correct: (Corrector) -> Void) {
		do {
			derivate(commandBuffer: commandBuffer, Δφ: Δφ, count: count, correct: correct)
		}
		do {
			assert( commandBuffer.device === derivatePipeline.P.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatePipeline.P.threadExecutionWidth
			encoder.setComputePipelineState(derivatePipeline.P)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(f, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.Derivate(\(count))"
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, correct: (Corrector) -> Void) {
		do {
			derivate(commandBuffer: commandBuffer, Δφ: Δφ, count: count, correct: correct)
		}
		do {
			assert( commandBuffer.device === derivatePipeline.V.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatePipeline.V.threadExecutionWidth
			encoder.setComputePipelineState(derivatePipeline.V)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(v, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.Derivate(\(count))"
			encoder.endEncoding()
		}
	}
}
extension DegenerateDistributor {
	private struct DegenerateConnector: Connector {
		let order: MTLCommandBuffer
		let state: ConnectPipeline
		let width: Int
		let refer: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		func connect(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.X.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.stride <= x.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.stride <= a.μ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.X.threadExecutionWidth
			encoder.setComputePipelineState(state.X)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(x, offset: 0, at: 1)
			encoder.setBuffer(a.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(refer))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ConnectX(\(width, refer))"
			encoder.endEncoding()
			
		}
		func connect(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer) {
			
			assert( order.device === state.A.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.stride <= a.μ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.stride <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.A.threadExecutionWidth
			encoder.setComputePipelineState(state.A)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(a.μ, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(refer))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ConnectA(\(width, refer))"
			encoder.endEncoding()
			
		}
		func connect(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
		}
		func connect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.stride <= c.μ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(c.μ, offset: 0, at: 1)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(refer))], length: MemoryLayout<uint2>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ConnectC(\(width, refer))"
			encoder.endEncoding()
			
		}
		func connect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(d, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(refer))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ConnectD(\(width, refer))"
			encoder.endEncoding()
			
		}
		func connect(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.E.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.stride <= Σ.μ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.stride <= d.length )
			assert( order.device === j.μ.device && width * refer * MemoryLayout<Float>.stride <= j.μ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.E.threadExecutionWidth
			encoder.setComputePipelineState(state.E)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(d, offset: 0, at: 2)
			encoder.setBuffer(j.μ, offset: 0, at: 3)
			encoder.setBytes([uint2(arrayLiteral: uint(width), uint(refer))], length: MemoryLayout<uint2>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.ConnectE(\(width, refer))"
			encoder.endEncoding()
			
		}
	}
	private func gradient(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		do {
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: count.rows * count.cols * MemoryLayout<Float>.stride), value: 0)
			encoder.label = "Degenerate.GradientJP(\(count))"
			encoder.endEncoding()
		}
		connect(DegenerateConnector(order: commandBuffer, state: connectPipeline, width: count.rows, refer: count.cols, Σ: j))
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		do {
			gradient(commandBuffer: commandBuffer, j: j, count: count, connect: connect)
		}
		do {
			assert( commandBuffer.device === gradientPipeline.JV.device )
			assert( commandBuffer.device === Δx.device && count.cols * MemoryLayout<Float>.stride <= Δx.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.JV.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.JV)
			encoder.setBuffer(Δx, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.cols), uint(count.rows))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.setThreadgroupMemoryLength(threads*MemoryLayout<float4>.stride, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (count.cols+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.GradientJV(\(count))"
			encoder.endEncoding()
		}
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		do {
			gradient(commandBuffer: commandBuffer, j: j, count: count, connect: connect)
		}
		do {
			assert( commandBuffer.device === gradientPipeline.GV.device )
			assert( commandBuffer.device === Δv.device && count.rows * MemoryLayout<Float>.stride <= Δv.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.GV)
			encoder.setBuffer(Δv, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.rows), uint(count.cols))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.GradientGV(\(count))"
			encoder.endEncoding()
		}
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), connect: (Connector)->Void) {
		do {
			gradient(commandBuffer: commandBuffer, j: j, count: count, connect: connect)
		}
		do {
			assert( commandBuffer.device === gradientPipeline.GP.device )
			assert( commandBuffer.device === Δθ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δθ.μ.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = gradientPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(gradientPipeline.GP)
			encoder.setBuffer(Δθ.μ, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.rows), uint(count.cols))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.label = "Degenerate.GradientGP(\(count))"
			encoder.endEncoding()
		}
	}
}
extension DegenerateDistributor: Distributor {
	
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
