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
	func collect(collector: Collector) {
		access {
			collector.collect(d: $0.μ, φ: cell.φ(-1))
		}
	}
}
extension Decay {
	func correct_refresh() {
		rotate()
	}
	func correct(commandBuffer: CommandBuffer, fix: Set<Cell>, Δφ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		if !fix.contains(cell) {
			change(commandBuffer: commandBuffer) {
				cell.distributor.derivate(commandBuffer: commandBuffer, Δv: $0.μ, j: j(0), Δφ: Δφ, φ: cell.φ(0), count: count) { jacobian in
					access {
						jacobian.jacobian(d: $0.μ, φ: cell.φ(-1))
					}
					cell.jacobian(jacobian: jacobian, feed: j)
				}
			}
		}
	}
}
extension Decay {
	func jacobian(jacobian: Jacobian, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		access {
			jacobian.jacobian(φ: cell.φ(-1), d: $0.μ, j: feed(-1))
		}
	}
}
extension Decay {
	@NSManaged private var cache: Array<Cache>
	@NSManaged private var index: Int
	private class Cache: NSObject {
		let j: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			j = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			super.init()
		}
		func reset(commandBuffer: CommandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[j.μ, j.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.label = "Decay.Cache.reset"
			encoder.endEncoding()
		}
	}
	internal func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].j
	}
	internal func rotate() {
		index = ( index + 1 ) % cache.count
	}
	override internal func setup(commandBuffer: CommandBuffer, count: Int) {
		cache = Array<Void>(repeating: (), count: cell.depth).map {
			Cache(context: context, count: count)
		}
		cache.forEach {
			$0.reset(commandBuffer: commandBuffer)
		}
		index = 0
		super.setup(commandBuffer: commandBuffer, count: count)
	}
}
extension Decay {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: cell.width)
		commandBuffer.label = "Decay(\(cell.label)).awakeFromFetch"
		commandBuffer.commit()
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: cell.width)
		commandBuffer.label = "Decay(\(cell.label)).awakeFromSnapshotEvents"
		commandBuffer.commit()
	}
}
extension Decay {
	@NSManaged var cell: Cell
}
extension Context {
	@nonobjc internal func make(commandBuffer: CommandBuffer, cell: Cell) throws -> Decay {
		let count: Int = cell.width
		let decay: Decay = try make()
		decay.cell = cell
		decay.locationType = AdapterType.Logistic.rawValue
		decay.location = Data(count: count * MemoryLayout<Float>.size)
		decay.location.withUnsafeMutableBytes {
			vDSP_vfill([0.0], $0, 1, vDSP_Length(count))
		}
		decay.scaleType = AdapterType.Discard.rawValue
		decay.scale = Data(count: count * MemoryLayout<Float>.size)
		decay.scale.withUnsafeMutableBytes {
			vDSP_vfill([0.0], $0, 1, vDSP_Length(count))
		}
		decay.setup(commandBuffer: commandBuffer, count: count)
		return decay
	}
}
