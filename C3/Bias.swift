//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import CoreData
import Distributor
internal class Bias: Arcane {
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
extension Bias {
	func collect_refresh() {
		guard cell.bias === self else { return }
	}
	func collect(collector: Collector) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		cell.distributor.flush(commandBuffer: collector.order, θ: cache[0].j)
		access(commandBuffer: collector.order) {
			cell.distributor.jacobian(commandBuffer: collector.order, Σ: cache[0].j, c: $0, count: count)
			collector.collect(c: $0)
		}
	}
}
extension Bias {
	func correct_refresh() {
		guard cell.bias === self else { return }
		cache.rotate()
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		cell.distributor.jacobian(commandBuffer: commandBuffer, j: cache[0].j, Σ: cache[0].j, φ: φ, count: count)
		update(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δ: $0, j: cache[0].j, Δφ: Δφ, count: count)
		}
	}
}
extension Bias {
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
extension Bias {
	@NSManaged var cell: Cell
}
extension Context {
	@nonobjc internal func make(commandBuffer: CommandBuffer, cell: Cell) throws -> Bias {
		typealias T = Float
		let count: Int = cell.width
		let bias: Bias = try make()
		let μ: T = T(0)
		let σ: T = T(1)
		bias.cell = cell
		bias.location = Data(bytes: Array<T>(repeating: μ, count: count), count: count * MemoryLayout<T>.size)
		bias.logscale = Data(bytes: Array<T>(repeating: σ, count: count), count: count * MemoryLayout<T>.size)
		bias.setup(commandBuffer: commandBuffer, count: count)
		return bias
	}
}
