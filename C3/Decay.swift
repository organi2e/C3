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
		custom = ( custom + 1 ) % cell.depth
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		change(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δv: $0.μ, j: j(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
				access {
					jacobian.jacobian(d: $0.μ, φ: cell.φ(-1))
				}
				cell.jacobian(jacobian: jacobian, feed: j)
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
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		do {
			let length: Int = count * MemoryLayout<Float>.size
			let ref: Array<Void> = Array<Void>(repeating: (), count: cell.depth)
			ju = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
			js = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
		}
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			(ju+js).forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
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
	@NSManaged var ju: Array<Buffer>
	@NSManaged var js: Array<Buffer>
	func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( ju.count == cell.depth )
		assert( js.count == cell.depth )
		return (μ: ju[((offset+custom)%ju.count+ju.count)%ju.count],
		        σ: js[((offset+custom)%js.count+js.count)%js.count])
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
