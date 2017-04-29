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
	let collectorPipeline: CollectorPipeline
	let correctorPipeline: CorrectorPipeline
	let activatorPipeline: ActivatorPipeline
	let derivatorPipeline: DerivatorPipeline
	let jacobianPipeline: JacobianPipeline
	public init(device: MTLDevice) throws {
		let bundle: Bundle = Bundle(for: DegenerateDistributor.self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		collectorPipeline = CollectorPipeline(
			W: try library.make(name: "DegenerateCollectW"),
			C: try library.make(name: "DegenerateCollectC"),
			D: try library.make(name: "DegenerateCollectD"),
			F: try library.make(name: "DegenerateCollectF")
		)
		correctorPipeline = CorrectorPipeline(
			J: try library.make(name: "DegenerateCorrectJ"),
			G: try library.make(name: "DegenerateCorrectG"),
			N: try library.make(name: "DegenerateCorrectN"),
			P: try library.make(name: "DegenerateCorrectP"),
			V: try library.make(name: "DegenerateCorrectV")
		)
		jacobianPipeline = JacobianPipeline(
			X: try library.make(name: "DegenerateJacobianX"),
			A: try library.make(name: "DegenerateJacobianA"),
			B: try library.make(name: "DegenerateJacobianB"),
			C: try library.make(name: "DegenerateJacobianC"),
			D: try library.make(name: "DegenerateJacobianD"),
			E: try library.make(name: "DegenerateJacobianE"),
			F: try library.make(name: "DegenerateJacobianF")
		)
		activatorPipeline = ActivatorPipeline(
			AP: try library.make(name: "DegenerateActivateP"),
			AV: try library.make(name: "DegenerateActivateV"),
			GP: try library.make(name: "DegenerateDerivateP"),
			GV: try library.make(name: "DegenerateDerivateV")
		)
		derivatorPipeline = DerivatorPipeline(
			JP: try library.make(name: "DegenerateDeltaJ"),
			JV: try library.make(name: "DegenerateDeltaJ"),
			GP: try library.make(name: "DegenerateDeltaG"),
			GV: try library.make(name: "DegenerateDeltaG")
		)
	}

}
extension DegenerateDistributor {
	private struct DegenerateCollector: Collector {
		let order: MTLCommandBuffer
		let state: CollectorPipeline
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
	private func collect(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector) -> Void) {
		do {
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.stride <= φ.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: φ.μ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.fill(buffer: φ.σ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.label = "Degenerate.CollectFlush(\(count))"
			encoder.endEncoding()
		}
		collector(DegenerateCollector(order: commandBuffer, state: collectorPipeline, width: count, Σ: φ))
	}
	public func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector) -> Void) {
		do {
			collect(commandBuffer: commandBuffer, φ: φ, count: count, collector: collector)
		}
		do {
			assert( commandBuffer.device === activatorPipeline.AP.device )
			assert( commandBuffer.device === f.device && count * MemoryLayout<Float>.stride <= f.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.AP.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.AP)
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
	public func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector) -> Void) {
		do {
			collect(commandBuffer: commandBuffer, φ: φ, count: count, collector: collector)
		}
		do {
			assert( commandBuffer.device === activatorPipeline.AV.device )
			assert( commandBuffer.device === v.device && count * MemoryLayout<Float>.stride <= v.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.AV.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.AV)
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
		let state: CorrectorPipeline
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
			encoder.endEncoding()
		}
		public func correct(φ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer) {
			assert( order.device === state.G.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.stride <= Δ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.stride <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.stride <= φ.σ.length )
			assert( order.device === v.device && width * MemoryLayout<Float>.stride <= v.length )
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.P)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(φ.μ, offset: 0, at: 1)
			encoder.setBuffer(φ.σ, offset: 0, at: 2)
			encoder.setBuffer(v, offset: 0, at: 3)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	private func correct(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector)->Void) {
		do {
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.stride <= Δφ.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δφ.μ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.fill(buffer: Δφ.σ, range: NSRange(location: 0, length: count * MemoryLayout<Float>.stride), value: 0)
			encoder.endEncoding()
		}
		corrector(DegenerateCorrector(order: commandBuffer, state: correctorPipeline, width: count, Δ: Δφ.μ))
	}
	public func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), f: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector) -> Void) {
		do {
			correct(commandBuffer: commandBuffer, Δφ: Δφ, count: count, corrector: corrector)
		}
		do {
			assert( commandBuffer.device === activatorPipeline.GP.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.GP)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(f, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector) -> Void) {
		do {
			correct(commandBuffer: commandBuffer, Δφ: Δφ, count: count, corrector: corrector)
		}
		do {
			assert( commandBuffer.device === activatorPipeline.GV.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.stride <= Δφ.μ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.stride <= g.μ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.stride <= φ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.GV)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(v, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension DegenerateDistributor {
	private struct DegenerateJacobian: Jacobian {
		let order: MTLCommandBuffer
		let state: JacobianPipeline
		let width: Int
		let refer: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		func jacobian(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer)) {
			
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
			encoder.endEncoding()
			
		}
		func jacobian(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer) {
			
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
			encoder.endEncoding()
			
		}
		func jacobian(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
		}
		func jacobian(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			
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
			encoder.endEncoding()
			
		}
		func jacobian(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			
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
			encoder.endEncoding()
			
		}
		func jacobian(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
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
			encoder.endEncoding()
			
		}
	}
	private func derivate(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.σ.length )
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: count.rows * count.cols * MemoryLayout<Float>.stride), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: count.rows * count.cols * MemoryLayout<Float>.stride), value: 0)
			encoder.endEncoding()
		}
		jacobian(DegenerateJacobian(order: commandBuffer, state: jacobianPipeline, width: count.rows, refer: count.cols, Σ: j))
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			derivate(commandBuffer: commandBuffer, j: j, count: count, jacobian: jacobian)
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.JV.device )
			assert( commandBuffer.device === Δx.device && count.cols * MemoryLayout<Float>.stride <= Δx.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.JV.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.JV)
			encoder.setBuffer(Δx, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.cols), uint(count.rows))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.setThreadgroupMemoryLength(threads*MemoryLayout<float4>.stride, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (count.cols+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			derivate(commandBuffer: commandBuffer, j: j, count: count, jacobian: jacobian)
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.GV.device )
			assert( commandBuffer.device === Δv.device && count.rows * MemoryLayout<Float>.stride <= Δv.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.GV)
			encoder.setBuffer(Δv, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.rows), uint(count.cols))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			derivate(commandBuffer: commandBuffer, j: j, count: count, jacobian: jacobian)
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.GP.device )
			assert( commandBuffer.device === Δθ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δθ.μ.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.stride <= j.μ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.stride <= Δφ.μ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.GP)
			encoder.setBuffer(Δθ.μ, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 2)
			encoder.setBytes([uint2(arrayLiteral: uint(count.rows), uint(count.cols))], length: MemoryLayout<uint2>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
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
