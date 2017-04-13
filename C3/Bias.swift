//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import CoreData
import Distributor
internal class Bias: Arcane {
	
}
extension Bias {
	func collect_refresh(commandBuffer: CommandBuffer) {
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector) {
		access {
			collector.collect(c: $0)
		}
	}
}
extension Bias {
	func correct_refresh() {
		custom = ( custom + 1 ) % cell.depth
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		change(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δθ: $0, j: j(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
				access {
					jacobian.jacobian(c: $0)
				}
				cell.jacobian(jacobian: jacobian, feed: j)
			}
		}
	}
}
extension Bias {
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
extension Bias {
	@NSManaged var ju: Array<Buffer>
	@NSManaged var js: Array<Buffer>
	func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( ju.count == cell.depth )
		assert( js.count == cell.depth )
		return (μ: ju[((offset+custom)%ju.count+ju.count)%ju.count],
		        σ: js[((offset+custom)%js.count+js.count)%js.count])
	}
}
extension Bias {
	@NSManaged var cell: Cell
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, cell: Cell, adapters: (AdapterType, AdapterType)) throws -> Bias {
		let count: Int = cell.width
		let bias: Bias = try make()
		bias.cell = cell
		bias.locationType = adapters.0.rawValue
		bias.location = Data(count: count * MemoryLayout<Float>.size)
		bias.location.withUnsafeMutableBytes {
			vDSP_vfill([0.0], $0, 1, vDSP_Length(count))
		}
		bias.scaleType = adapters.1.rawValue
		bias.scale = Data(count: count * MemoryLayout<Float>.size)
		bias.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		bias.setup(commandBuffer: commandBuffer, count: count)
		return bias
	}
}
