//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import CoreData
import Distributor
internal class Edge: Arcane {
	
}
extension Edge {
	func collect_refresh(commandBuffer: CommandBuffer) {
		input.collect_refresh(commandBuffer: commandBuffer)
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector, visit: Set<Cell>) {
		let x: Buffer = input.collect(visit: visit)
		access {
			collector.collect(w: $0, x: x, count: input.width)
		}
	}
}
extension Edge {
	func correct_refresh() {
		output.correct_refresh()
		rotate()
	}
	func correct(corrector: Corrector, fix: Set<Cell>, visit: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let Δφ: (μ: Buffer, σ: Buffer) = output.correct(fix: fix, visit: visit)
		let φ: (μ: Buffer, σ: Buffer) = output.φ(0)
		if !fix.contains(output) {
			change(commandBuffer: corrector.order) {
				output.distributor.derivate(commandBuffer: corrector.order, Δθ: $0, j: ja(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
					access {
						jacobian.jacobian(a: $0, x: input.χ(0))
					}
					output.jacobian(jacobian: jacobian, feed: ja)
				}
			}
		}
		output.distributor.derivate(commandBuffer: corrector.order, Δx: corrector.Δ, j: jx(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
			access {
				jacobian.jacobian(x: input.χ(0), a: $0)
			}
			output.jacobian(jacobian: jacobian, feed: jx)
		}
	}
}
extension Edge {
	@NSManaged private var cache: Array<Cache>
	@NSManaged private var index: Int
	private class Cache: NSObject {
		let ja: (μ: Buffer, σ: Buffer)
		let jx: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			ja = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			jx = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			super.init()
		}
		func reset(commandBuffer: CommandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[ja.μ, ja.σ, jx.μ, jx.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
	}
	internal func ja(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].ja
	}
	internal func jx(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].jx
	}
	internal func rotate() {
		index = ( index + 1 ) % cache.count
	}
	override internal func setup(commandBuffer: CommandBuffer, count: Int) {
		cache = Array<Void>(repeating: (), count: output.depth).map {
			Cache(context: context, count: count)
		}
		cache.forEach {
			$0.reset(commandBuffer: commandBuffer)
		}
		index = 0
		super.setup(commandBuffer: commandBuffer, count: count)
	}
}
extension Edge {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: output.width * input.width)
		commandBuffer.commit()
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: output.width * input.width)
		commandBuffer.commit()
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, output: Cell, input: Cell, adapters: (AdapterType, AdapterType)) throws -> Edge {
		let count: Int = output.width * input.width
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.locationType = adapters.0.rawValue
		edge.location = Data(count: count * MemoryLayout<Float>.size)
		edge.location.withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) -> Void in
			assert( MemoryLayout<Float>.size == 4 )
			assert( MemoryLayout<UInt32>.size == 4 )
			arc4random_buf(ref, edge.location.count)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
			cblas_sscal(Int32(count/2), 2*Float.pi, ref.advanced(by: count/2), 1)
			vvlogf(ref, ref, [Int32(count/2)])
			cblas_sscal(Int32(count/2), -2, ref, 1)
			vvsqrtf(ref, ref, [Int32(count/2)])
			vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: count/2), 2, vDSP_Length(count/4))
			vDSP_rect(ref, 2, ref, 2, vDSP_Length(count/2))
		}
		edge.scaleType = adapters.1.rawValue
		edge.scale = Data(count: count * MemoryLayout<Float>.size)
		edge.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		edge.setup(commandBuffer: commandBuffer, count: count)
		return edge
	}
}
