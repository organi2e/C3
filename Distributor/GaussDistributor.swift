//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import Metal
/*
public class GaussActivator {
	let width: Int
	var index: Int
	let potential: Array<(μ: MTLBuffer, σ: MTLBuffer)>
	let gradients: Array<(μ: MTLBuffer, σ: MTLBuffer)>
	let activatorPipeline: MTLComputePipelineState
	let derivatorPipeline: MTLComputePipelineState
	let collectorPipeline: CollectorPipeline
	let correctorPipeline: CorrectorPipeline
	private init(device: MTLDevice, pipeline: (
		activator: MTLComputePipelineState,
		derivator: MTLComputePipelineState,
		collector: CollectorPipeline,
		corrector: CorrectorPipeline),
	             count: Int,
	             depth: Int) {
		width = count
		index = 0
		do {
			let options: MTLResourceOptions = .storageModeShared
			let length: Int = width * MemoryLayout<Float>.size
			potential = Array<Void>(repeating: (), count: depth).map {(
				μ: device.makeBuffer(length: length, options: options),
				σ: device.makeBuffer(length: length, options: options)
				)}
			gradients = Array<Void>(repeating: (), count: depth).map {(
				μ: device.makeBuffer(length: length, options: options),
				σ: device.makeBuffer(length: length, options: options)
				)}
		}
		activatorPipeline = pipeline.activator
		derivatorPipeline = pipeline.derivator
		collectorPipeline = pipeline.collector
		correctorPipeline = pipeline.corrector
	}
	public static func factory(device: MTLDevice) throws -> (Int, Int) -> Activator {
		let bundle: Bundle = Bundle(for: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let activateP: MTLComputePipelineState = try library.make(name: "GaussActivateP")
		let derivateP: MTLComputePipelineState = try library.make(name: "GaussDerivateP")
		let collector: CollectorPipeline = CollectorPipeline(
			W: try library.make(name: "GaussCollectW"),
			C: try library.make(name: "GaussCollectC"),
			D: try library.make(name: "GaussCollectD"),
			F: try library.make(name: "GaussCollectF")
		)
		let corrector: CorrectorPipeline = CorrectorPipeline(
			J: try library.make(name: "GaussCorrectJ"),
			G: try library.make(name: "GaussCorrectG"),
			D: try library.make(name: "GaussCorrectD")
		)
		return {
			return GaussActivator(device: device,
			                      pipeline: (
									activator: activateP,
									derivator: derivateP,
									collector: collector,
									corrector: corrector),
			                      count: $0.0,
			                      depth: $0.1)
		}
	}
}
extension GaussActivator {
	private struct GaussCollector: Collector {
		let order: MTLCommandBuffer
		let state: CollectorPipeline
		let width: Int
		let φ: (μ: MTLBuffer, σ: MTLBuffer)
		public func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			assert( order.device === state.W.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === w.μ.device && width * count * MemoryLayout<Float>.size <= w.μ.length )
			assert( order.device === w.σ.device && width * count * MemoryLayout<Float>.size <= w.σ.length )
			assert( order.device === x.device && count * MemoryLayout<Float>.size <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.W.threadExecutionWidth
			encoder.setComputePipelineState(state.W)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(w.μ, offset: 0, at: 2)
			encoder.setBuffer(w.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(count)], length: 2*MemoryLayout<Float>.size, at: 5)
			encoder.setThreadgroupMemoryLength(2*4*threads*MemoryLayout<Float>.size, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func collect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
			
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
			encoder.endEncoding()
		}
		public func collect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
			
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
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, f: MTLBuffer, collector: (Collector)->Void) {
		index = index + 1
		let Φ: (μ: MTLBuffer, σ: MTLBuffer) = potential[index%potential.count]
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Φ.μ, range: NSRange(location: 0, length: Φ.μ.length), value: 0)
			encoder.fill(buffer: Φ.σ, range: NSRange(location: 0, length: Φ.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			collector(GaussCollector(order: commandBuffer, state: collectorPipeline, width: width, Σ: Φ))
		}
		do {
			assert( commandBuffer.device === collectorPipeline.F.device )
			assert( commandBuffer.device === Φ.μ.device && width * MemoryLayout<Float>.size <= Φ.μ.length )
			assert( commandBuffer.device === Φ.σ.device && width * MemoryLayout<Float>.size <= Φ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = collectorPipeline).threadExecutionWidth
			encoder.setComputePipelineState(collectorPipeline.F)
			encoder.setBuffer(Φ.μ, offset: 0, at: 0)
			encoder.setBuffer(Φ.σ, offset: 0, at: 1)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === activatorPipeline.device )
			assert( commandBuffer.device === f.device && width * MemoryLayout<Float>.size <= f.length )
			assert( commandBuffer.device === Φ.μ.device && width * MemoryLayout<Float>.size <= Φ.μ.length )
			assert( commandBuffer.device === Φ.σ.device && width * MemoryLayout<Float>.size <= Φ.σ.length )
			
			typealias T = uint
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.threadExecutionWidth
			let seeds: Data = Data(capacity: width * MemoryLayout<T>.size)
			encoder.setComputePipelineState(activatorPipeline)
			encoder.setBuffer(f, offset: 0, at: 0)
			encoder.setBuffer(Φ.μ, offset: 0, at: 1)
			encoder.setBuffer(Φ.σ, offset: 0, at: 2)
			arc4random_buf(UnsafeMutablePointer(mutating: seeds.withUnsafeBytes{$0}), seeds.count)
			seeds.withUnsafeBytes {
				encoder.setBytes($0, length: seeds.count, at: 3)
			}
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension GaussActivator {
	private struct GaussCorrector: Corrector {
		let order: MTLCommandBuffer
		let state: CorrectorPipeline
		let width: Int
		let Σ: MTLBuffer
		public func correct(χ: MTLBuffer, ϝ: MTLBuffer) {
			
		}
		public func correct(j: (μ: MTLBuffer, σ: MTLBuffer), Δ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
			
			assert( order.device === state.J.device )
			assert( order.device === Σ.device && width * MemoryLayout<Float>.size <= Σ.length )
			assert( order.device === j.μ.device && width * count * MemoryLayout<Float>.size <= j.μ.length )
			assert( order.device === j.σ.device && width * count * MemoryLayout<Float>.size <= j.σ.length )
			assert( order.device === Δ.μ.device && count * MemoryLayout<Float>.size <= Δ.μ.length )
			assert( order.device === Δ.σ.device && count * MemoryLayout<Float>.size <= Δ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.J.threadExecutionWidth
			encoder.setComputePipelineState(state.J)
			encoder.setBuffer(Σ, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(count)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.setThreadgroupMemoryLength(4*threads*MemoryLayout<Float>.size, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func correct(Δ: MTLBuffer) {
			
			assert( order.device === state.G.device )
			assert( order.device === Σ.device && width * MemoryLayout<Float>.size <= Σ.length )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.size <= Δ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.G)
			encoder.setBuffer(Σ, offset: 0, at: 0)
			encoder.setBuffer(Δ, offset: 0, at: 1)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), corrector: (Corrector)->Void) {
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = potential[index%potential.count]
		let g: (μ: MTLBuffer, σ: MTLBuffer) = gradients[index%gradients.count]
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: g.μ, range: NSRange(location: 0, length: g.μ.length), value: 0)
			//encoder.fill(buffer: g.σ, range: NSRange(location: 0, length: g.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			corrector(GaussCorrector(order: commandBuffer, state: correctorPipeline, width: width, Σ: g.μ))
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.device )
			assert( commandBuffer.device === Δφ.μ.device && width * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && width * MemoryLayout<Float>.size <= Δφ.σ.length )
			assert( commandBuffer.device === g.μ.device && width * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && width * MemoryLayout<Float>.size <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
			assert( commandBuffer.device === g.μ.device && width * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && width * MemoryLayout<Float>.size <= g.σ.length )
			
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(g.σ, offset: 0, at: 3)
			encoder.setBuffer(φ.μ, offset: 0, at: 4)
			encoder.setBuffer(φ.σ, offset: 0, at: 5)
			encoder.setBuffer(g.μ, offset: 0, at: 6)
			encoder.setBuffer(g.σ, offset: 0, at: 7)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 8)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension GaussActivator {
	public var φ: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) {
		let buffer: Array<(μ: MTLBuffer, σ: MTLBuffer)> = potential
		let offset: Int = index
		return {
			return buffer[((offset+$0)%buffer.count+buffer.count)%buffer.count]
		}
	}
	public var g: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) {
		let buffer: Array<(μ: MTLBuffer, σ: MTLBuffer)> = gradients
		let offset: Int = index
		return {
			return buffer[((offset+$0)%buffer.count+buffer.count)%buffer.count]
		}
	}
}
extension GaussActivator: Activator {
	
}
*/
public class GaussDerivator {
	let width: Int
	let refer: Int
	var index: Int
	var fixed: Bool
	let jacobMatrices: Array<(μ: MTLBuffer, σ: MTLBuffer)>
	let jacobPipeline: JacobianPipeline
	let deltaPipeline: DerivatorPipeline
	private init(device: MTLDevice,
	             pipeline: (J: JacobianPipeline, D: DerivatorPipeline),
	             count: (width: Int, refer: Int),
	             depth: Int) {
		width = count.width
		refer = count.refer
		index = 0
		fixed = true
		do {
			let options: MTLResourceOptions = .storageModeShared
			let length: Int = width * refer * MemoryLayout<Float>.size
			jacobMatrices = Array<Void>(repeating: (), count: depth).map {(
				μ: device.makeBuffer(length: length, options: options),
				σ: device.makeBuffer(length: length, options: options)
			)}
		}
		jacobPipeline = pipeline.J
		deltaPipeline = pipeline.D
	}
	public static func factory(device: MTLDevice) throws -> (Int, Int, Int) -> Derivator {
		let bundle: Bundle = Bundle(for: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let J: JacobianPipeline = JacobianPipeline(
			X: try library.make(name: "GaussJacobianX"),
			A: try library.make(name: "GaussJacobianA"),
			B: try library.make(name: "GaussJacobianB"),
			C: try library.make(name: "GaussJacobianC"),
			D: try library.make(name: "GaussJacobianD"),
			E: try library.make(name: "GaussJacobianE"),
			F: try library.make(name: "GaussJacobianF")
		)
		let D: DerivatorPipeline = DerivatorPipeline(
			JP: try library.make(name: "GaussDeltaJP"),
			JV: try library.make(name: "GaussDeltaJV"),
			GP: try library.make(name: "GaussDeltaGP"),
			GV: try library.make(name: "GaussDeltaGV")
		)
		return {
			return GaussDerivator(device: device,
			                      pipeline: (J: J, D: D),
			                      count: ($0.0, $0.1),
			                      depth: $0.2)
		}
	}
}
extension GaussDerivator {
	public func flush(commandBuffer: MTLCommandBuffer) {
		index = ( index + 1 ) % jacobMatrices.count
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert(commandBuffer.device === j.μ.device)
		assert(commandBuffer.device === j.σ.device)
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
		encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
		encoder.endEncoding()
		fixed = false
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer)) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.X.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === x.device && refer * MemoryLayout<Float>.size <= x.length )
		assert( commandBuffer.device === a.μ.device && refer * MemoryLayout<Float>.size <= a.μ.length )
		assert( commandBuffer.device === a.σ.device && refer * MemoryLayout<Float>.size <= a.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.X.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.X)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(x, offset: 0, at: 2)
		encoder.setBuffer(a.μ, offset: 0, at: 3)
		encoder.setBuffer(a.σ, offset: 0, at: 4)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.A.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === a.μ.device && width * refer * MemoryLayout<Float>.size <= a.μ.length )
		assert( commandBuffer.device === a.σ.device && width * refer * MemoryLayout<Float>.size <= a.σ.length )
		assert( commandBuffer.device === x.device && refer * MemoryLayout<Float>.size <= x.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.A.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.A)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(a.μ, offset: 0, at: 2)
		encoder.setBuffer(a.σ, offset: 0, at: 3)
		encoder.setBuffer(x, offset: 0, at: 4)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.B.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === b.μ.device && width * width * MemoryLayout<Float>.size <= b.μ.length )
		assert( commandBuffer.device === b.σ.device && width * width * MemoryLayout<Float>.size <= b.σ.length )
		assert( commandBuffer.device === y.device && width * MemoryLayout<Float>.size <= y.length )
		assert( commandBuffer.device === g.μ.device && width * MemoryLayout<Float>.size <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && width * MemoryLayout<Float>.size <= g.σ.length )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let block: Int = 8
		encoder.setComputePipelineState(jacobPipeline.B)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(b.μ, offset: 0, at: 2)
		encoder.setBuffer(b.σ, offset: 0, at: 3)
		encoder.setBuffer(y, offset: 0, at: 4)
		encoder.setBuffer(g.μ, offset: 0, at: 5)
		encoder.setBuffer(g.σ, offset: 0, at: 6)
		encoder.setBuffer(j.μ, offset: 0, at: 7)
		encoder.setBuffer(j.σ, offset: 0, at: 8)
		encoder.setBytes([uint(width), uint(refer), uint(width), uint(block)], length: 4*MemoryLayout<uint>.size, at: 9)
		encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/4/block+1, height: (refer-1)/4/block+1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: block, height: block, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, c: (μ: MTLBuffer, σ: MTLBuffer)) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.C.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
		assert( commandBuffer.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.C.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.C)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(c.μ, offset: 0, at: 2)
		encoder.setBuffer(c.σ, offset: 0, at: 3)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index%jacobMatrices.count]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.D.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === d.device && width * MemoryLayout<Float>.size <= d.length )
		assert( commandBuffer.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.D.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.D)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(d, offset: 0, at: 2)
		encoder.setBuffer(φ.μ, offset: 0, at: 3)
		encoder.setBuffer(φ.σ, offset: 0, at: 4)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.E.device )
		assert( commandBuffer.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === d.device && width * MemoryLayout<Float>.size <= d.length )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.E.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.E)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(d, offset: 0, at: 2)
		encoder.setBuffer(φ.μ, offset: 0, at: 3)
		encoder.setBuffer(φ.σ, offset: 0, at: 4)
		encoder.setBuffer(j.μ, offset: 0, at: 5)
		encoder.setBuffer(j.σ, offset: 0, at: 6)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 7)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func fix(commandBuffer: MTLCommandBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( !fixed )
		assert( commandBuffer.device === jacobPipeline.F.device )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobPipeline.F.threadExecutionWidth
		encoder.setComputePipelineState(jacobPipeline.F)
		encoder.setBuffer(j.μ, offset: 0, at: 0)
		encoder.setBuffer(j.σ, offset: 0, at: 1)
		encoder.setBuffer(φ.μ, offset: 0, at: 2)
		encoder.setBuffer(φ.σ, offset: 0, at: 3)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<Float>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
		fixed = true
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( fixed )
		assert( commandBuffer.device === deltaPipeline.JV.device )
		assert( commandBuffer.device === Δx.device && refer * MemoryLayout<Float>.size <= Δx.length )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Δφ.μ.device && refer * MemoryLayout<Float>.size <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && refer * MemoryLayout<Float>.size <= Δφ.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = deltaPipeline.JV.threadExecutionWidth
		encoder.setComputePipelineState(deltaPipeline.JV)
		encoder.setBuffer(Δx, offset: 0, at: 0)
		encoder.setBuffer(j.μ, offset: 0, at: 1)
		encoder.setBuffer(j.σ, offset: 0, at: 2)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
		encoder.setBytes([uint(refer), uint(width)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.setThreadgroupMemoryLength(4*threads*MemoryLayout<Float>.size, at: 0)
		encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( fixed )
		assert( commandBuffer.device === deltaPipeline.GV )
		assert( commandBuffer.device === Δv.device && width * refer * MemoryLayout<Float>.size <= Δv.length )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Δφ.μ.device && width * MemoryLayout<Float>.size <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && width * MemoryLayout<Float>.size <= Δφ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = deltaPipeline.GV.threadExecutionWidth
		encoder.setComputePipelineState(deltaPipeline.GV)
		encoder.setBuffer(Δv, offset: 0, at: 0)
		encoder.setBuffer(j.μ, offset: 0, at: 1)
		encoder.setBuffer(j.σ, offset: 0, at: 2)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacobMatrices[index]
		assert( fixed )
		assert( commandBuffer.device === deltaPipeline.GP.device )
		assert( commandBuffer.device === Δθ.μ.device && width * MemoryLayout<Float>.size <= Δθ.μ.length )
		assert( commandBuffer.device === Δθ.σ.device && width * MemoryLayout<Float>.size <= Δθ.σ.length )
		assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Δφ.μ.device && width * MemoryLayout<Float>.size <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && width * MemoryLayout<Float>.size <= Δφ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = deltaPipeline.GP.threadExecutionWidth
		encoder.setComputePipelineState(deltaPipeline.GP)
		encoder.setBuffer(Δθ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δθ.σ, offset: 0, at: 1)
		encoder.setBuffer(j.μ, offset: 0, at: 2)
		encoder.setBuffer(j.σ, offset: 0, at: 3)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 4)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 5)
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 6)
		encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension GaussDerivator: Derivator {
	public var j: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) {
		let buffer: Array<(μ: MTLBuffer, σ: MTLBuffer)> = jacobMatrices
		let offset: Int = index
		return {
			return buffer[((offset+$0)%buffer.count+buffer.count)%buffer.count]
		}
	}
}
public class GaussDistributor {
	let collectorPipeline: CollectorPipeline
	let correctorPipeline: CorrectorPipeline
	let activatorPipeline: ActivatorPipeline
	let derivatorPipeline: DerivatorPipeline
	let jacobianPipeline: JacobianPipeline
	public init(device: MTLDevice, xorshift: (Int, Int, Int) = (5, 7, 4)) throws {
		let bundle: Bundle = Bundle(for: GaussDistributor.self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let xorshiftValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		xorshiftValues.setConstantValue([uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)], type: .uint3, withName: "xorshift")
		collectorPipeline = CollectorPipeline(
			W: try library.make(name: "GaussCollectW"),
			C: try library.make(name: "GaussCollectC"),
			D: try library.make(name: "GaussCollectD"),
			F: try library.make(name: "GaussCollectF")
		)
		correctorPipeline = CorrectorPipeline(
			J: try library.make(name: "GaussCorrectJ"),
			G: try library.make(name: "GaussCorrectG"),
			N: try library.make(name: "GaussCorrectN")
		)
		jacobianPipeline = JacobianPipeline(
			X: try library.make(name: "GaussJacobianX"),
			A: try library.make(name: "GaussJacobianA"),
			B: try library.make(name: "GaussJacobianB"),
			C: try library.make(name: "GaussJacobianC"),
			D: try library.make(name: "GaussJacobianD"),
			E: try library.make(name: "GaussJacobianE"),
			F: try library.make(name: "GaussJacobianF")
		)
		activatorPipeline = ActivatorPipeline(
			AP: try library.make(name: "GaussActivateP", constantValues: xorshiftValues),
			AV: try library.make(name: "GaussActivateV", constantValues: xorshiftValues),
			GP: try library.make(name: "GaussDerivateP"),
			GV: try library.make(name: "GaussDerivateV")
		)
		derivatorPipeline = DerivatorPipeline(
			JP: try library.make(name: "GaussDeltaJP"),
			JV: try library.make(name: "GaussDeltaJV"),
			GP: try library.make(name: "GaussDeltaGP"),
			GV: try library.make(name: "GaussDeltaGV")
		)
	}
}
extension GaussDistributor {
	private struct GaussCollector: Collector {
		let order: MTLCommandBuffer
		let state: CollectorPipeline
		let width: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		public func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			
			assert( order.device === state.W.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === w.μ.device && width * count * MemoryLayout<Float>.size <= w.μ.length )
			assert( order.device === w.σ.device && width * count * MemoryLayout<Float>.size <= w.σ.length )
			assert( order.device === x.device && count * MemoryLayout<Float>.size <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.W.threadExecutionWidth
			encoder.setComputePipelineState(state.W)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(w.μ, offset: 0, at: 2)
			encoder.setBuffer(w.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(count)], length: 2*MemoryLayout<Float>.size, at: 5)
			encoder.setThreadgroupMemoryLength(2*4*threads*MemoryLayout<Float>.size, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (width+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func collect(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
			
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
			encoder.endEncoding()
		}
		public func collect(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
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
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, p: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector) -> Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: φ.μ, range: NSRange(location: 0, length: φ.μ.length), value: 0)
			encoder.fill(buffer: φ.σ, range: NSRange(location: 0, length: φ.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			collector(GaussCollector(order: commandBuffer, state: collectorPipeline, width: count, Σ: φ))
		}
		do {
			assert( commandBuffer.device === collectorPipeline.F.device )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = collectorPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(collectorPipeline.F)
			encoder.setBuffer(φ.μ, offset: 0, at: 0)
			encoder.setBuffer(φ.σ, offset: 0, at: 1)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === activatorPipeline.AP.device )
			assert( commandBuffer.device === p.device && count * MemoryLayout<Float>.size <= p.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.size <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			typealias T = ushort
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.AP.threadExecutionWidth
			let bytes: Int = 4 * threads * MemoryLayout<T>.size
			let data: Data = Data(capacity: bytes)
			encoder.setComputePipelineState(activatorPipeline.AP)
			encoder.setBuffer(p, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(g.σ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			arc4random_buf(UnsafeMutablePointer(mutating: data.withUnsafeBytes{$0}), bytes)
			data.withUnsafeBytes {
				encoder.setBytes($0, length: bytes, at: 5)
			}
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, collector: (Collector) -> Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: φ.μ, range: NSRange(location: 0, length: φ.μ.length), value: 0)
			encoder.fill(buffer: φ.σ, range: NSRange(location: 0, length: φ.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			collector(GaussCollector(order: commandBuffer, state: collectorPipeline, width: count, Σ: φ))
		}
		do {
			assert( commandBuffer.device === collectorPipeline.F.device )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = collectorPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(collectorPipeline.F)
			encoder.setBuffer(φ.μ, offset: 0, at: 0)
			encoder.setBuffer(φ.σ, offset: 0, at: 1)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === activatorPipeline.AV.device )
			assert( commandBuffer.device === v.device && count * MemoryLayout<Float>.size <= v.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.size <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			typealias T = ushort
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.AV.threadExecutionWidth
			let bytes: Int = 4 * threads * MemoryLayout<T>.size
			let data: Data = Data(capacity: bytes)
			encoder.setComputePipelineState(activatorPipeline.AV)
			encoder.setBuffer(v, offset: 0, at: 0)
			encoder.setBuffer(g.μ, offset: 0, at: 1)
			encoder.setBuffer(g.σ, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			arc4random_buf(UnsafeMutablePointer(mutating: data.withUnsafeBytes{$0}), bytes)
			data.withUnsafeBytes {
				encoder.setBytes($0, length: bytes, at: 5)
			}
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension GaussDistributor {
	private struct GaussCorrector: Corrector {
		let order: MTLCommandBuffer
		let state: CorrectorPipeline
		let width: Int
		let Δ: MTLBuffer
		public func correct(χ: MTLBuffer, ϝ: MTLBuffer) {
			assert( order.device === state.G.device )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.size <= Δ.length)
			assert( order.device === χ.device && width * MemoryLayout<Float>.size <= χ.length)
			assert( order.device === ϝ.device && width * MemoryLayout<Float>.size <= ϝ.length)
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
			assert( order.device === Δ.device && width * MemoryLayout<Float>.size <= Δ.length)
			assert( order.device === χ.device && width * MemoryLayout<Float>.size <= χ.length)
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
	}
	public func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), p: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector) -> Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δφ.μ, range: NSRange(location: 0, length: Δφ.μ.length), value: 0)
			encoder.fill(buffer: Δφ.σ, range: NSRange(location: 0, length: Δφ.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			corrector(GaussCorrector(order: commandBuffer, state: correctorPipeline, width: count, Δ: Δφ.μ))
		}
		do {
			assert( commandBuffer.device === activatorPipeline.GP.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.size <= Δφ.σ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.size <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.GP)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(g.σ, offset: 0, at: 3)
			encoder.setBuffer(φ.μ, offset: 0, at: 4)
			encoder.setBuffer(φ.σ, offset: 0, at: 5)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func activate(commandBuffer: MTLCommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), v: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: Int, corrector: (Corrector) -> Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δφ.μ, range: NSRange(location: 0, length: Δφ.μ.length), value: 0)
			encoder.fill(buffer: Δφ.σ, range: NSRange(location: 0, length: Δφ.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			corrector(GaussCorrector(order: commandBuffer, state: correctorPipeline, width: count, Δ: Δφ.μ))
		}
		do {
			assert( commandBuffer.device === activatorPipeline.GV.device )
			assert( commandBuffer.device === Δφ.μ.device && count * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count * MemoryLayout<Float>.size <= Δφ.σ.length )
			assert( commandBuffer.device === g.μ.device && count * MemoryLayout<Float>.size <= g.μ.length )
			assert( commandBuffer.device === g.σ.device && count * MemoryLayout<Float>.size <= g.σ.length )
			assert( commandBuffer.device === φ.μ.device && count * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = activatorPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(activatorPipeline.GV)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 1)
			encoder.setBuffer(g.μ, offset: 0, at: 2)
			encoder.setBuffer(g.σ, offset: 0, at: 3)
			encoder.setBuffer(φ.μ, offset: 0, at: 4)
			encoder.setBuffer(φ.σ, offset: 0, at: 5)
			encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension GaussDistributor {
	private struct GaussJacobian: Jacobian {
		let order: MTLCommandBuffer
		let state: JacobianPipeline
		let width: Int
		let refer: Int
		let Σ: (μ: MTLBuffer, σ: MTLBuffer)
		func jacobian(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.X.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.size <= x.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.size <= a.μ.length )
			assert( order.device === a.σ.device && width * refer * MemoryLayout<Float>.size <= a.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.X.threadExecutionWidth
			encoder.setComputePipelineState(state.X)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBuffer(a.μ, offset: 0, at: 3)
			encoder.setBuffer(a.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
		func jacobian(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer) {
			
			assert( order.device === state.A.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.size <= a.μ.length )
			assert( order.device === a.σ.device && width * refer * MemoryLayout<Float>.size <= a.σ.length )
			assert( order.device === x.device   && refer * MemoryLayout<Float>.size <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.A.threadExecutionWidth
			encoder.setComputePipelineState(state.A)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(a.μ, offset: 0, at: 2)
			encoder.setBuffer(a.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
		func jacobian(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
		}
		func jacobian(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.C.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(c.μ, offset: 0, at: 2)
			encoder.setBuffer(c.σ, offset: 0, at: 3)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
		func jacobian(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.D.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(Σ.μ, offset: 0, at: 0)
			encoder.setBuffer(Σ.σ, offset: 0, at: 1)
			encoder.setBuffer(d, offset: 0, at: 2)
			encoder.setBuffer(φ.μ, offset: 0, at: 3)
			encoder.setBuffer(φ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
		func jacobian(φ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.E.device )
			assert( order.device === Σ.μ.device && width * refer * MemoryLayout<Float>.size <= Σ.μ.length )
			assert( order.device === Σ.σ.device && width * refer * MemoryLayout<Float>.size <= Σ.σ.length )
			assert( order.device === φ.μ.device && width * MemoryLayout<Float>.size <= φ.μ.length )
			assert( order.device === φ.σ.device && width * MemoryLayout<Float>.size <= φ.σ.length )
			assert( order.device === d.device   && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
			assert( order.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
			
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
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 7)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δx: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			jacobian(GaussJacobian(order: commandBuffer, state: jacobianPipeline, width: count.rows, refer: count.cols, Σ: j))
		}
		do {
			assert( commandBuffer.device === jacobianPipeline.F.device )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === φ.μ.device && count.rows * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count.rows * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = jacobianPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(jacobianPipeline.F)
			encoder.setBuffer(j.μ, offset: 0, at: 0)
			encoder.setBuffer(j.σ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.σ, offset: 0, at: 3)
			encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth : 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.JV.device )
			assert( commandBuffer.device === Δx.device && count.cols * MemoryLayout<Float>.size <= Δx.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.size <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.JV.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.JV)
			encoder.setBuffer(Δx, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(count.cols), uint(count.rows)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.setThreadgroupMemoryLength(4*threads*MemoryLayout<Float>.size, at: 0)
			encoder.dispatchThreadgroups(MTLSize(width: (count.cols+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δv: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			jacobian(GaussJacobian(order: commandBuffer, state: jacobianPipeline, width: count.rows, refer: count.cols, Σ: j))
		}
		do {
			assert( commandBuffer.device === jacobianPipeline.F.device )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === φ.μ.device && count.rows * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count.rows * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = jacobianPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(jacobianPipeline.F)
			encoder.setBuffer(j.μ, offset: 0, at: 0)
			encoder.setBuffer(j.σ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.σ, offset: 0, at: 3)
			encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth : 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.GV.device )
			assert( commandBuffer.device === Δv.device && count.rows * MemoryLayout<Float>.size <= Δv.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.size <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.GV.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.GV)
			encoder.setBuffer(Δv, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δθ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer),
	                     Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), jacobian: (Jacobian)->Void) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			jacobian(GaussJacobian(order: commandBuffer, state: jacobianPipeline, width: count.rows, refer: count.cols, Σ: j))
		}
		do {
			assert( commandBuffer.device === jacobianPipeline.F.device )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === φ.μ.device && count.rows * MemoryLayout<Float>.size <= φ.μ.length )
			assert( commandBuffer.device === φ.σ.device && count.rows * MemoryLayout<Float>.size <= φ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = jacobianPipeline.F.threadExecutionWidth
			encoder.setComputePipelineState(jacobianPipeline.F)
			encoder.setBuffer(j.μ, offset: 0, at: 0)
			encoder.setBuffer(j.σ, offset: 0, at: 1)
			encoder.setBuffer(φ.μ, offset: 0, at: 2)
			encoder.setBuffer(φ.σ, offset: 0, at: 3)
			encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth : 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === derivatorPipeline.GP.device )
			assert( commandBuffer.device === Δθ.μ.device && count.rows * MemoryLayout<Float>.size <= Δθ.μ.length )
			assert( commandBuffer.device === Δθ.σ.device && count.rows * MemoryLayout<Float>.size <= Δθ.σ.length )
			assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.size <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = derivatorPipeline.GP.threadExecutionWidth
			encoder.setComputePipelineState(derivatorPipeline.GP)
			encoder.setBuffer(Δθ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δθ.σ, offset: 0, at: 1)
			encoder.setBuffer(j.μ, offset: 0, at: 2)
			encoder.setBuffer(j.σ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 4)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 5)
			encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 6)
			encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension GaussDistributor {
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.X.device )
		assert( commandBuffer.device === Σ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === x.device && count.cols * MemoryLayout<Float>.size <= x.length )
		assert( commandBuffer.device === a.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.μ.length )
		assert( commandBuffer.device === a.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.X.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.X)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(x, offset: 0, at: 2)
		encoder.setBuffer(a.μ, offset: 0, at: 3)
		encoder.setBuffer(a.σ, offset: 0, at: 4)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.A.device )
		assert( commandBuffer.device === Σ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === a.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.μ.length )
		assert( commandBuffer.device === a.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.σ.length )
		assert( commandBuffer.device === x.device && count.cols * MemoryLayout<Float>.size <= x.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.A.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.A)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(a.μ, offset: 0, at: 2)
		encoder.setBuffer(a.σ, offset: 0, at: 3)
		encoder.setBuffer(x, offset: 0, at: 4)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.B.device )
		assert( commandBuffer.device === Σ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === b.μ.device && count.rows * count.rows * MemoryLayout<Float>.size <= b.μ.length )
		assert( commandBuffer.device === b.σ.device && count.rows * count.rows * MemoryLayout<Float>.size <= b.σ.length )
		assert( commandBuffer.device === y.device && count.rows * MemoryLayout<Float>.size <= y.length )
		assert( commandBuffer.device === g.μ.device && count.rows * MemoryLayout<Float>.size <= g.μ.length )
		assert( commandBuffer.device === g.σ.device && count.rows * MemoryLayout<Float>.size <= g.σ.length )
		assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let block: Int = 8
		encoder.setComputePipelineState(jacobianPipeline.B)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(b.μ, offset: 0, at: 2)
		encoder.setBuffer(b.σ, offset: 0, at: 3)
		encoder.setBuffer(y, offset: 0, at: 4)
		encoder.setBuffer(g.μ, offset: 0, at: 5)
		encoder.setBuffer(g.σ, offset: 0, at: 6)
		encoder.setBuffer(j.μ, offset: 0, at: 7)
		encoder.setBuffer(j.σ, offset: 0, at: 8)
		encoder.setBytes([uint(count.rows), uint(count.cols), uint(count.rows), uint(block)], length: 4*MemoryLayout<uint>.size, at: 9)
		encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/4/block+1, height: (count.cols-1)/4/block+1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: block, height: block, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.C.device )
		assert( commandBuffer.device === Σ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === c.μ.device && count.rows * MemoryLayout<Float>.size <= c.μ.length )
		assert( commandBuffer.device === c.σ.device && count.rows * MemoryLayout<Float>.size <= c.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.C.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.C)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(c.μ, offset: 0, at: 2)
		encoder.setBuffer(c.σ, offset: 0, at: 3)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.D.device )
		assert( commandBuffer.device === Σ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( commandBuffer.device === d.device && count.rows * MemoryLayout<Float>.size <= d.length )
		assert( commandBuffer.device === φ.μ.device && count.rows * MemoryLayout<Float>.size <= φ.μ.length )
		assert( commandBuffer.device === φ.σ.device && count.rows * MemoryLayout<Float>.size <= φ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.D.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.D)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(d, offset: 0, at: 2)
		encoder.setBuffer(φ.μ, offset: 0, at: 3)
		encoder.setBuffer(φ.σ, offset: 0, at: 4)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.E.device )
		assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Σ.μ.device && count.rows * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === Σ.μ.device && count.rows * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === φ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === φ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.E.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.E)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(φ.μ, offset: 0, at: 2)
		encoder.setBuffer(φ.σ, offset: 0, at: 3)
		encoder.setBuffer(d, offset: 0, at: 4)
		encoder.setBuffer(j.μ, offset: 0, at: 5)
		encoder.setBuffer(j.σ, offset: 0, at: 6)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<Float>.size, at: 7)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), Σ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === jacobianPipeline.F.device )
		assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Σ.μ.device && count.rows * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === Σ.σ.device && count.rows * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === φ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === φ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = jacobianPipeline.F.threadExecutionWidth
		encoder.setComputePipelineState(jacobianPipeline.F)
		encoder.setBuffer(j.μ, offset: 0, at: 0)
		encoder.setBuffer(j.σ, offset: 0, at: 1)
		encoder.setBuffer(Σ.μ, offset: 0, at: 2)
		encoder.setBuffer(Σ.σ, offset: 0, at: 3)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<Float>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension GaussDistributor {
	public func derivate(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === derivatorPipeline.GV.device )
		assert( commandBuffer.device === Δ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.length )
		assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.size <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.size <= Δφ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = derivatorPipeline.GV.threadExecutionWidth
		encoder.setComputePipelineState(derivatorPipeline.GV)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(j.μ, offset: 0, at: 1)
		encoder.setBuffer(j.σ, offset: 0, at: 2)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<Float>.size, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		assert( commandBuffer.device === derivatorPipeline.GP.device )
		assert( commandBuffer.device === Δ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.μ.length )
		assert( commandBuffer.device === Δ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.σ.length )
		assert( commandBuffer.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( commandBuffer.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
		assert( commandBuffer.device === Δφ.μ.device && count.rows * MemoryLayout<Float>.size <= Δφ.μ.length )
		assert( commandBuffer.device === Δφ.σ.device && count.rows * MemoryLayout<Float>.size <= Δφ.σ.length )
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let threads: Int = derivatorPipeline.GP.threadExecutionWidth
		encoder.setComputePipelineState(derivatorPipeline.GP)
		encoder.setBuffer(Δ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δ.σ, offset: 0, at: 1)
		encoder.setBuffer(j.μ, offset: 0, at: 2)
		encoder.setBuffer(j.σ, offset: 0, at: 3)
		encoder.setBuffer(Δφ.μ, offset: 0, at: 4)
		encoder.setBuffer(Δφ.σ, offset: 0, at: 5)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<Float>.size, at: 6)
		encoder.dispatchThreadgroups(MTLSize(width: (count.rows-1)/threads+1, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension GaussDistributor {
	public func flush(commandBuffer: MTLCommandBuffer, θ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: θ.μ, range: NSRange(location: 0, length: θ.μ.length), value: 0)
		encoder.fill(buffer: θ.σ, range: NSRange(location: 0, length: θ.σ.length), value: 0)
		encoder.endEncoding()
	}
	public func flush(commandBuffer: MTLCommandBuffer, θ: MTLBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: θ, range: NSRange(location: 0, length: θ.length), value: 0)
		encoder.endEncoding()
	}
}
extension GaussDistributor: Distributor {

}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
