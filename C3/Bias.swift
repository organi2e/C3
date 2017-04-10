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
extension Bias {
	func collect_refresh(commandBuffer: CommandBuffer) {
		guard cell.bias.objectID == objectID else { fatalError() }
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector) {
		guard cell.bias.objectID == objectID else { fatalError() }
		access {
			collector.collect(c: $0)
		}
	}
}
extension Bias {
	func correct_refresh() {
		guard cell.bias.objectID == objectID else { fatalError() }
		cache.rotate()
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		change(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δθ: $0, j: cache[0].j, Δφ: Δφ, φ: φ, count: count) { jacobian in
				access {
					jacobian.jacobian(c: $0)
				}
				cell.jacobian(jacobian: jacobian) {
					cache[$0].j
				}
			}
		}
	}
}
extension Bias {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		let ref: Array<Void> = Array<Void>(repeating: (), count: cell.depth)
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
