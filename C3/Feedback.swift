//
//  Feedback.swift
//  macOS
//
//  Created by Kota Nakano on 4/8/17.
//
//
import Accelerate
import CoreData
import Metal
import Distributor
internal class Feedback: Arcane {
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
extension Feedback {
	func collect_refresh(commandBuffer: CommandBuffer) {
		guard cell.loop.contains(self) else { fatalError() }
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector) {
		guard cell.loop.contains(self) else { fatalError() }
		access {
			collector.collect(w: $0, x: cell.cache[depth].χ, count: cell.width)
		}
	}
}
extension Feedback {
	func correct_refresh() {
		guard cell.loop.contains(self) else { fatalError() }
		cache.rotate()
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: cell.width)
		change(commandBuffer: commandBuffer) {
			cell.distributor.derivate(commandBuffer: commandBuffer, Δθ: $0, j: cache[0].j, Δφ: Δφ, φ: φ, count: count) { jacobian in
				access {
					jacobian.jacobian(a: $0, x: cell.cache[depth].χ)
				}
				cell.jacobian(jacobian: jacobian) {
					cache[$0].j
				}
			}
		}
	}
}
extension Feedback {
	func jacobian(jacobian: Jacobian, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		guard cell.loop.contains(self) else { fatalError() }
		access {
			jacobian.jacobian(b: $0, y: cell.cache[depth].χ, g: cell.cache[depth].g, j: feed(depth))
		}
	}
}
extension Feedback {
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
		setup(commandBuffer: commandBuffer, count: cell.width * cell.width)
		commandBuffer.commit()
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: cell.width * cell.width)
		commandBuffer.commit()
	}
}
extension Feedback {
	@NSManaged var cell: Cell
	@NSManaged var depth: Int
}
extension Context {
	func make(commandBuffer: CommandBuffer, cell: Cell, depth: Int) throws -> Feedback {
		let count: Int = cell.width * cell.width
		let feedback: Feedback = try make()
		feedback.cell = cell
		feedback.depth = depth
		feedback.location = Data(count: count * MemoryLayout<Float>.size)
		feedback.scale = Data(count: count * MemoryLayout<Float>.size)
		feedback.location.withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) -> Void in
			assert( MemoryLayout<Float>.size == 4 )
			assert( MemoryLayout<UInt32>.size == 4 )
			arc4random_buf(ref, feedback.location.count)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
			cblas_sscal(Int32(count/2), 2*Float.pi, ref.advanced(by: count/2), 1)
			vvlogf(ref, ref, [Int32(count/2)])
			cblas_sscal(Int32(count/2), -2, ref, 1)
			vvsqrtf(ref, ref, [Int32(count/2)])
			vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: count/2), 2, vDSP_Length(count/4))
			vDSP_rect(ref, 2, ref, 2, vDSP_Length(count/2))
		}
		feedback.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		feedback.setup(commandBuffer: commandBuffer, count: count)
		return feedback
	}
}
