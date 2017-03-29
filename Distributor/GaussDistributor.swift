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
			G: try library.make(name: "GaussCorrectG")
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
		let value: (μ: MTLBuffer, σ: MTLBuffer)
		public func collect(w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			assert( order.device === state.W.device )
			assert( order.device === value.μ.device && width * MemoryLayout<Float>.size <= value.μ.length )
			assert( order.device === value.σ.device && width * MemoryLayout<Float>.size <= value.σ.length )
			assert( order.device === w.μ.device && width * count * MemoryLayout<Float>.size <= w.μ.length )
			assert( order.device === w.σ.device && width * count * MemoryLayout<Float>.size <= w.σ.length )
			assert( order.device === x.device && count * MemoryLayout<Float>.size <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.W.threadExecutionWidth
			encoder.setComputePipelineState(state.W)
			encoder.setBuffer(value.μ, offset: 0, at: 0)
			encoder.setBuffer(value.σ, offset: 0, at: 1)
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
			assert( order.device === value.μ.device && width * MemoryLayout<Float>.size <= value.μ.length )
			assert( order.device === value.σ.device && width * MemoryLayout<Float>.size <= value.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(value.μ, offset: 0, at: 0)
			encoder.setBuffer(value.σ, offset: 0, at: 1)
			encoder.setBuffer(c.μ, offset: 0, at: 2)
			encoder.setBuffer(c.σ, offset: 0, at: 3)
			encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func collect(d: MTLBuffer, Φ: (μ: MTLBuffer, σ: MTLBuffer)) {
			assert( order.device === state.D.device )
			assert( order.device === value.μ.device && width * MemoryLayout<Float>.size <= value.μ.length )
			assert( order.device === value.σ.device && width * MemoryLayout<Float>.size <= value.σ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === Φ.μ.device && width * MemoryLayout<Float>.size <= Φ.μ.length )
			assert( order.device === Φ.σ.device && width * MemoryLayout<Float>.size <= Φ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(value.μ, offset: 0, at: 0)
			encoder.setBuffer(value.σ, offset: 0, at: 1)
			encoder.setBuffer(d, offset: 0, at: 2)
			encoder.setBuffer(Φ.μ, offset: 0, at: 3)
			encoder.setBuffer(Φ.σ, offset: 0, at: 4)
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
			collector(GaussCollector(order: commandBuffer, state: collectorPipeline, width: width, value: Φ))
		}
		do {
			assert( commandBuffer.device === collectorPipeline.F.device )
			assert( commandBuffer.device === Φ.μ.device && width * MemoryLayout<Float>.size <= Φ.μ.length )
			assert( commandBuffer.device === Φ.σ.device && width * MemoryLayout<Float>.size <= Φ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = collectorPipeline.F.threadExecutionWidth
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
			let bytes: Int = width * MemoryLayout<T>.size
			let seeds: Data = Data(capacity: bytes)
			encoder.setComputePipelineState(activatorPipeline)
			encoder.setBuffer(f, offset: 0, at: 0)
			encoder.setBuffer(Φ.μ, offset: 0, at: 1)
			encoder.setBuffer(Φ.σ, offset: 0, at: 2)
			seeds.withUnsafeBytes {
				arc4random_buf(UnsafeMutablePointer(mutating: $0), bytes)
				//vDSP_vfilli([-1], UnsafeMutablePointer(mutating: $0), 1, vDSP_Length(bytes/MemoryLayout<Int32>.size))
			}
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
		let error: MTLBuffer
		public func correct(j: (μ: MTLBuffer, σ: MTLBuffer), Δ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
			
			assert( order.device === state.J.device )
			assert( order.device === error.device && width * MemoryLayout<Float>.size <= error.length )
			assert( order.device === j.μ.device && width * count * MemoryLayout<Float>.size <= j.μ.length )
			assert( order.device === j.σ.device && width * count * MemoryLayout<Float>.size <= j.σ.length )
			assert( order.device === Δ.μ.device && count * MemoryLayout<Float>.size <= Δ.μ.length )
			assert( order.device === Δ.σ.device && count * MemoryLayout<Float>.size <= Δ.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.J.threadExecutionWidth
			encoder.setComputePipelineState(state.J)
			encoder.setBuffer(error, offset: 0, at: 0)
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
			assert( order.device === error.device && width * MemoryLayout<Float>.size <= error.length )
			assert( order.device === Δ.device && width * MemoryLayout<Float>.size <= Δ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.G.threadExecutionWidth
			encoder.setComputePipelineState(state.G)
			encoder.setBuffer(error, offset: 0, at: 0)
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
			corrector(GaussCorrector(order: commandBuffer, state: correctorPipeline, width: width, error: g.μ))
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
public class GaussDerivator {
	let width: Int
	let refer: Int
	var index: Int
	let jacob: Array<(μ: MTLBuffer, σ: MTLBuffer)>
	let jacobPipeline: JacobianPipeline
	let paramPipeline: MTLComputePipelineState
	let valuePipeline: MTLComputePipelineState
	private init(device: MTLDevice, pipeline: (
		P: MTLComputePipelineState,
		V: MTLComputePipelineState,
		J: JacobianPipeline),
	             count: (width: Int, refer: Int),
	             depth: Int) {
		width = count.width
		refer = count.refer
		index = 0
		do {
			let options: MTLResourceOptions = .storageModeShared
			let length: Int = width * refer * MemoryLayout<Float>.size
			jacob = Array<Void>(repeating: (), count: depth).map {(
				μ: device.makeBuffer(length: length, options: options),
				σ: device.makeBuffer(length: length, options: options)
				)}
		}
		jacobPipeline = pipeline.J
		paramPipeline = pipeline.P
		valuePipeline = pipeline.V
	}
	public static func factory(device: MTLDevice) throws -> (Int, Int, Int) -> Derivator {
		let bundle: Bundle = Bundle(for: self)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		let GP: MTLComputePipelineState = try library.make(name: "GaussDeltaGP")
		let GV: MTLComputePipelineState = try library.make(name: "GaussDeltaGV")
		let J: JacobianPipeline = JacobianPipeline(
			X: try library.make(name: "GaussJacobianX"),
			Y: try library.make(name: "GaussJacobianX"),
			A: try library.make(name: "GaussJacobianA"),
			B: try library.make(name: "GaussJacobianB"),
			C: try library.make(name: "GaussJacobianC"),
			D: try library.make(name: "GaussJacobianD"),
			F: try library.make(name: "GaussJacobianF")
		)
		return {
			return GaussDerivator(device: device,
			                      pipeline: (
									P: GP,
									V: GV,
									J: J),
			                      count: ($0.0, $0.1),
			                      depth: $0.2)
		}
	}
}
extension GaussDerivator {
	private struct GaussJacobian: Jacobian {
		let order: MTLCommandBuffer
		let state: JacobianPipeline
		let width: Int
		let refer: Int
		let jacob: (μ: MTLBuffer, σ: MTLBuffer)
		public func jacobian(x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
			
			assert( order.device === state.X.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === x.device && refer * MemoryLayout<Float>.size <= x.length )
			assert( order.device === a.μ.device && refer * MemoryLayout<Float>.size <= a.μ.length )
			assert( order.device === a.σ.device && refer * MemoryLayout<Float>.size <= a.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.X.threadExecutionWidth
			encoder.setComputePipelineState(state.X)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBuffer(a.μ, offset: 0, at: 3)
			encoder.setBuffer(a.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
			
		}
		public func jacobian(y: MTLBuffer, b: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.X.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === y.device && width * MemoryLayout<Float>.size <= y.length )
			assert( order.device === b.μ.device && refer * MemoryLayout<Float>.size <= b.μ.length )
			assert( order.device === b.σ.device && refer * MemoryLayout<Float>.size <= b.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.X.threadExecutionWidth
			encoder.setComputePipelineState(state.X)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
			encoder.setBuffer(y, offset: 0, at: 2)
			encoder.setBuffer(b.μ, offset: 0, at: 3)
			encoder.setBuffer(b.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func jacobian(a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
			
			assert( order.device === state.A.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === a.μ.device && width * refer * MemoryLayout<Float>.size <= a.μ.length )
			assert( order.device === a.σ.device && width * refer * MemoryLayout<Float>.size <= a.σ.length )
			assert( order.device === x.device && refer * MemoryLayout<Float>.size <= x.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.A.threadExecutionWidth
			encoder.setComputePipelineState(state.A)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
			encoder.setBuffer(a.μ, offset: 0, at: 2)
			encoder.setBuffer(a.σ, offset: 0, at: 3)
			encoder.setBuffer(x, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func jacobian(b: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.B.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === b.μ.device && width * width * MemoryLayout<Float>.size <= b.μ.length )
			assert( order.device === b.σ.device && width * width * MemoryLayout<Float>.size <= b.σ.length )
			assert( order.device === y.device && width * MemoryLayout<Float>.size <= y.length )
			assert( order.device === g.μ.device && width * MemoryLayout<Float>.size <= g.μ.length )
			assert( order.device === g.σ.device && width * MemoryLayout<Float>.size <= g.σ.length )
			assert( order.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
			assert( order.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let block: Int = 8
			encoder.setComputePipelineState(state.B)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
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
			encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 2)
			encoder.setThreadgroupMemoryLength(4*4*block*block*MemoryLayout<Float>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/4/block+1, height: (refer-1)/4/block+1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: block, height: block, depth: 1))
			encoder.endEncoding()
		}
		public func jacobian(c: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.C.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === c.μ.device && width * MemoryLayout<Float>.size <= c.μ.length )
			assert( order.device === c.σ.device && width * MemoryLayout<Float>.size <= c.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.C.threadExecutionWidth
			encoder.setComputePipelineState(state.C)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
			encoder.setBuffer(c.μ, offset: 0, at: 2)
			encoder.setBuffer(c.σ, offset: 0, at: 3)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		public func jacobian(d: MTLBuffer, φ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer)) {
			
			assert( order.device === state.D.device )
			assert( order.device === jacob.μ.device && width * refer * MemoryLayout<Float>.size <= jacob.μ.length )
			assert( order.device === jacob.σ.device && width * refer * MemoryLayout<Float>.size <= jacob.σ.length )
			assert( order.device === d.device && width * MemoryLayout<Float>.size <= d.length )
			assert( order.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
			assert( order.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
			
			let encoder: MTLComputeCommandEncoder = order.makeComputeCommandEncoder()
			let threads: Int = state.D.threadExecutionWidth
			encoder.setComputePipelineState(state.D)
			encoder.setBuffer(jacob.μ, offset: 0, at: 0)
			encoder.setBuffer(jacob.σ, offset: 0, at: 1)
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
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), jacobian: (Jacobian)->Void) {
		index = index + 1
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacob[index%jacob.count]
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			jacobian(GaussJacobian(order: commandBuffer, state: jacobPipeline, width: width, refer: refer, jacob: j))
		}
		do {
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
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === valuePipeline.device )
			assert( commandBuffer.device === Δ.device && width * refer * MemoryLayout<Float>.size <= Δ.length )
			assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && width * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && width * MemoryLayout<Float>.size <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = valuePipeline.threadExecutionWidth
			encoder.setComputePipelineState(valuePipeline)
			encoder.setBuffer(Δ, offset: 0, at: 0)
			encoder.setBuffer(j.μ, offset: 0, at: 1)
			encoder.setBuffer(j.σ, offset: 0, at: 2)
			encoder.setBuffer(Δφ.μ, offset: 0, at: 3)
			encoder.setBuffer(Δφ.σ, offset: 0, at: 4)
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func derivate(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer), jacobian: (Jacobian)->Void) {
		index = index + 1
		let j: (μ: MTLBuffer, σ: MTLBuffer) = jacob[index%jacob.count]
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: j.μ, range: NSRange(location: 0, length: j.μ.length), value: 0)
			encoder.fill(buffer: j.σ, range: NSRange(location: 0, length: j.σ.length), value: 0)
			encoder.endEncoding()
		}
		do {
			jacobian(GaussJacobian(order: commandBuffer, state: jacobPipeline, width: width, refer: refer, jacob: j))
		}
		do {
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
			encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/threads+1, height: refer, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			assert( commandBuffer.device === paramPipeline.device )
			assert( commandBuffer.device === Δ.μ.device && width * MemoryLayout<Float>.size <= Δ.μ.length )
			assert( commandBuffer.device === Δ.σ.device && width * MemoryLayout<Float>.size <= Δ.σ.length )
			assert( commandBuffer.device === j.μ.device && width * refer * MemoryLayout<Float>.size <= j.μ.length )
			assert( commandBuffer.device === j.σ.device && width * refer * MemoryLayout<Float>.size <= j.σ.length )
			assert( commandBuffer.device === Δφ.μ.device && width * MemoryLayout<Float>.size <= Δφ.μ.length )
			assert( commandBuffer.device === Δφ.σ.device && width * MemoryLayout<Float>.size <= Δφ.σ.length )
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let threads: Int = paramPipeline.threadExecutionWidth
			encoder.setComputePipelineState(paramPipeline)
			encoder.setBuffer(Δ.μ, offset: 0, at: 0)
			encoder.setBuffer(Δ.σ, offset: 0, at: 1)
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
}
extension GaussDerivator: Derivator {
	public var j: (Int) -> (μ: MTLBuffer, σ: MTLBuffer) {
		let buffer: Array<(μ: MTLBuffer, σ: MTLBuffer)> = jacob
		let offset: Int = index
		return {
			return buffer[((offset+$0)%buffer.count+buffer.count)%buffer.count]
		}
	}
}
public class GaussDerivatorBTPP {
	
}
public class GaussDerivatorRTRL {
	
}
private extension MTLLibrary {
	func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
