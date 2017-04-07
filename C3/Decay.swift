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
		init(context: Context, count: Int, encoder: BlitCommandEncoder) {
			let length: Int = count * MemoryLayout<Float>.size
			j = (
				μ: context.make(length: length, options: .storageModePrivate),
				σ: context.make(length: length, options: .storageModePrivate)
			)
			[j.μ, j.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
		}
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
}
extension Decay {
	func collect_refresh() {
		
	}
	func collect(collector: Collector) {
		guard cell.decay?.objectID == objectID else { fatalError() }
		access(commandBuffer: collector.order) {
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
		update(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δv: $0.σ, j: cache[0].j, Δφ: Δφ, φ: φ, count: count) {(jacobian: Jacobian)in
				access(commandBuffer: commandBuffer) {
					jacobian.jacobian(d: $0.σ, φ: cell.cache[-1].φ)
				}
				cell.jacobian(jacobian: jacobian, j: cache[-1].j)
			}
		}
	}
}
extension Decay {
	func jacobian(jacobian: Jacobian, j: (μ: Buffer, σ: Buffer)) {
		access(commandBuffer: jacobian.order) {
			jacobian.jacobian(φ: cell.cache[-1].φ, d: $0.σ, j: j)
		}
	}
}
extension Decay {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		let ref: Array<Void> = Array<Void>(repeating: (), count: count)
		cache = RingBuffer<Cache>(buffer: ref.map {
			Cache(context: context, count: count, encoder: encoder)
		}, offset: 0)
		encoder.endEncoding()
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
