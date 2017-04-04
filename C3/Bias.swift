//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Distributor
internal class Bias: Arcane {
	var j: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: [], offset: 0)
}
extension Bias {
	func collect(collector: Collector) {
		let commandBuffer: CommandBuffer = collector.order
		refresh(commandBuffer: commandBuffer) {
			cell.distributor.flush(commandBuffer: commandBuffer, θ: j.rotate())
			cell.distributor.jacobian(commandBuffer: commandBuffer, Σ: j[0], c: $0, count: (rows: cell.width, cols: 1))
			collector.collect(c: $0)
		}
	}
}
extension Bias {
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		cell.distributor.jacobian(commandBuffer: commandBuffer, j: j[0], Σ: j[0], φ: φ, count: count)
		update(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δ: $0, j: j[0], Δφ: Δφ, count: count)
		}
	}
}
extension Bias {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		j = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: Array<Void>(repeating: (), count: cell.depth).map {(
			μ: context.make(length: cell.width * MemoryLayout<Float>.size),
			σ: context.make(length: cell.width * MemoryLayout<Float>.size)
		)}, offset: 0)
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
	internal func make(commandBuffer: CommandBuffer, cell: Cell) throws -> Bias {
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
