//
//  Decay.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//

import Accelerate
import Distributor
import CoreData

internal class Decay: Arcane {
	
}
extension Decay {
	func collect_refresh(commandBuffer: CommandBuffer) {
		fixing(commandBuffer: commandBuffer)
	}
	func collect(commandBuffer: CommandBuffer, collector: Collector) {
		access {
			collector.collect(d: $0.μ, φ: cell.φ(-1))
		}
	}
}
extension Decay {
	func correct_refresh(commandBuffer: CommandBuffer) {
		rotate()
	}
	func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>, Δφ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		if !ignore.contains(cell) {
			change(commandBuffer: commandBuffer) {
				cell.distributor.gradient(commandBuffer: commandBuffer, Δv: $0.μ, j: j(0), Δφ: Δφ, φ: cell.φ(0), count: count) { connector in
					access {
						connector.connect(d: $0.μ, φ: cell.φ(-1))
					}
					cell.connect(connector: connector, feed: j)
				}
			}
		}
	}
}
extension Decay {
	func connect(connector: Connector, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		access {
			connector.connect(φ: cell.φ(-1), d: $0.μ, j: feed(-1))
		}
	}
}
extension Decay {
	@NSManaged private var cache: Cache
	private class Cache: NSObject {
		var index: Int
		let array: Array<(μ: Buffer, σ: Buffer)>
		init(context: Context, depth: Int, width: Int) {
			let length: Int = width * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			index = 0
			array = Array<Void>(repeating: (), count: depth)
				.map{
					(μ: context.make(length: length, options: option),
					 σ: context.make(length: length, options: option))
			}
			super.init()
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			array
				.map{[$0.μ, $0.σ]}
				.reduce([], +)
				.forEach{encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)}
			encoder.label = "Decay.Cache.reset"
			encoder.endEncoding()
			commandBuffer.label = "Decay.Cache.reset"
			commandBuffer.commit()
		}
	}
	internal func rotate() {
		cache.index = ( cache.index + 1 ) % cache.array.count
	}
	internal func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle]
	}
	override internal func setup(context: Context, count: Int) {
		cache = Cache(context: context, depth: cell.depth, width: count)
		super.setup(context: context, count: count)
	}
}
extension Decay {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		try?eval {
			setup(context: $0, count: cell.width)
		}
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		try?eval {
			setup(context: $0, count: cell.width)
		}
	}
}
extension Decay {
	@NSManaged var cell: Cell
}
extension Context {
	@nonobjc internal func make(cell: Cell) throws -> Decay {
		let count: Int = cell.width
		let decay: Decay = try make()
		decay.cell = cell
		decay.locationType = AdapterType.Logistic.rawValue
		decay.location = Data(count: count * MemoryLayout<Float>.size)
		decay.location.withUnsafeMutableBytes {
			vDSP_vfill([Float(0)], $0, 1, vDSP_Length(count))
		}
		decay.scaleType = AdapterType.Discard.rawValue
		decay.scale = Data(count: count * MemoryLayout<Float>.size)
		decay.scale.withUnsafeMutableBytes {
			vDSP_vfill([Float(0)], $0, 1, vDSP_Length(count))
		}
		decay.setup(context: self, count: count)
		return decay
	}
}
