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
	func collect(commandBuffer: CommandBuffer, collector: Collector, visit: Set<Cell>) {
		let x: Buffer = input.collect(commandBuffer: commandBuffer, visit: visit)
		access {
			collector.collect(w: $0, x: x, count: input.width)
		}
	}
}
extension Edge {
	func correct_refresh(commandBuffer: CommandBuffer) {
		output.correct_refresh(commandBuffer: commandBuffer)
		rotate()
	}
	func correct(commandBuffer: CommandBuffer, corrector: Corrector, ignore: Set<Cell>, visit: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let Δφ: (μ: Buffer, σ: Buffer) = output.correct(commandBuffer: commandBuffer, ignore: ignore, visit: visit)
		let φ: (μ: Buffer, σ: Buffer) = output.φ(0)
		if !ignore.contains(output) {
			change(commandBuffer: corrector.order) {
				output.distributor.gradient(commandBuffer: corrector.order, Δθ: $0, j: ja(0), Δφ: Δφ, φ: φ, count: count) { connector in
					access {
						connector.connect(a: $0, x: input.χ(0))
					}
					output.connect(connector: connector, feed: ja)
				}
			}
		}
		output.distributor.gradient(commandBuffer: corrector.order, Δx: corrector.Δ, j: jx(0), Δφ: Δφ, φ: φ, count: count) { connector in
			access {
				connector.connect(x: input.χ(0), a: $0)
			}
			output.connect(connector: connector, feed: jx)
		}
	}
}
extension Edge {
	@NSManaged private var cache: Cache
	private class Cache: NSObject {
		var index: Int
		let array: Array<(ja: (μ: Buffer, σ: Buffer), jx: (μ: Buffer, σ: Buffer))>
		init(context: Context, depth: Int, width: Int) {
			let length: Int = width * MemoryLayout<Float>.stride
			let option: MTLResourceOptions = .storageModeShared
			index = 0
			array = Array<Void>(repeating: (), count: depth)
				.map{
					(ja: (μ: context.make(length: length, options: option), σ: context.make(length: length, options: option)),
					 jx: (μ: context.make(length: length, options: option), σ: context.make(length: length, options: option)))
			}
			super.init()
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			array
				.map{[$0.ja.μ, $0.ja.σ, $0.jx.μ, $0.jx.σ]}
				.reduce([], +)
				.forEach{encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)}
			encoder.label = "Edge.Cache.reset"
			encoder.endEncoding()
			commandBuffer.label = "Edge.Cache.reset"
			commandBuffer.commit()
		}
	}
	internal func rotate() {
		cache.index = ( cache.index + 1 ) % cache.array.count
	}
	internal func ja(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].ja
	}
	internal func jx(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].jx
	}
	override internal func setup(context: Context, count: Int) {
		cache = Cache(context: context, depth: output.depth, width: count)
		super.setup(context: context, count: count)
	}
}
extension Edge {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		try?eval {
			setup(context: $0, count: output.width * input.width)
		}
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		try?eval {
			setup(context: $0, count: output.width * input.width)
		}
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	internal func make(output: Cell, input: Cell, adapters: (AdapterType, AdapterType)) throws -> Edge {
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
			vDSP_vfill([Float(1)], $0, 1, vDSP_Length(count))
		}
		edge.setup(context: self, count: count)
		return edge
	}
}
