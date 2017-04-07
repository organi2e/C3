//
//  Decay.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//

import Distributor

internal class Decay: Arcane {
	struct Cache {
		let j: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			j = (
				μ: context.make(length: length, options: .storageModePrivate),
				σ: context.make(length: length, options: .storageModePrivate)
			)
		}
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
}
extension Decay {
	func collect_refresh(commandBuffer: CommandBuffer) {
		guard cell.decay?.objectID == objectID else { fatalError() }
		refresh(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		cell.distributor.flush(commandBuffer: collector.order, θ: cache[0].j)
		access(commandBuffer: collector.order) {
			cell.distributor.jacobian(commandBuffer: collector.order, Σ: cache[0].j, d: $0.σ, φ: cell.cache[-1].φ, count: count)
			collector.collect(d: $0.σ, φ: cell.cache[-1].φ)
		}
	}
}
extension Decay {
	func correct_refresh() {
		guard cell.decay?.objectID == objectID else { fatalError() }
		cache.rotate()
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		cell.jacobian(commandBuffer: commandBuffer, Σ: cache[0].j, j: cache[-1].j, count: count)
		cell.distributor.jacobian(commandBuffer: commandBuffer, j: cache[0].j, Σ: cache[0].j, φ: φ, count: count)
		update(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δ: $0.σ, j: cache[0].j, Δφ: Δφ, count: count)
		}
	}
}
extension Decay {
	func jacobian(commandBuffer: CommandBuffer, Σ: (μ: Buffer, σ: Buffer), j: (μ: Buffer, σ: Buffer), count: (rows: Int, cols: Int)) {
		access(commandBuffer: commandBuffer) {
			cell.distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, d: $0.σ, φ: cell.cache[-1].φ, j: j, count: count)
		}
	}
}
extension Decay {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		let ref: Array<Void> = Array<Void>(repeating: (), count: count)
		cache = RingBuffer<Cache>(buffer: ref.map {
			Cache(context: context, count: count)
		}, offset: 0)
	}
	override func awakeFromFetch() {
		super.awakeFromFetch()
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: cell.width)
		commandBuffer.commit()
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: cell.width)
		commandBuffer.commit()
	}
}
extension Decay {
	@NSManaged var cell: Cell
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, cell: Cell) throws -> Decay {
		typealias T = Float
		let count: Int = cell.width
		let decay: Decay = try make()
		decay.cell = cell
		decay.location = Data(count: count * MemoryLayout<Float>.size)
		decay.locationType = AdapterType.Linear.rawValue
		decay.scale = Data(count: count * MemoryLayout<Float>.size)
		decay.scaleType = AdapterType.Logistic.rawValue
		decay.setup(commandBuffer: commandBuffer, count: count)
		return decay
	}
}
