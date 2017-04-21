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
	
}
extension Feedback {
	func collect_refresh(commandBuffer: CommandBuffer) {
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector) {
		access {
			collector.collect(w: $0, x: cell.χ(refer), count: cell.width)
		}
	}
}
extension Feedback {
	func correct_refresh() {
		custom = ( custom + 1 ) % cell.depth
	}
	func correct(commandBuffer: CommandBuffer, Δφ: (μ: MTLBuffer, σ: MTLBuffer), φ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: cell.width)
		if cell.pliable {
			change(commandBuffer: commandBuffer) {
				cell.distributor.derivate(commandBuffer: commandBuffer, Δθ: $0, j: j(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
					access {
						jacobian.jacobian(a: $0, x: cell.χ(refer))
					}
					cell.jacobian(jacobian: jacobian, feed: j)
				}
			}
		}
	}
}
extension Feedback {
	func jacobian(jacobian: Jacobian, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		access {
			jacobian.jacobian(b: $0, y: cell.χ(refer), g: cell.g(refer), j: feed(refer))
		}
	}
}
extension Feedback {
	@NSManaged private var cache: Array<Cache>
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
			encoder.endEncoding()
		}
	}
	func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+custom)%cycle+cycle)%cycle].j
	}
	override internal func setup(commandBuffer: CommandBuffer, count: Int) {
		cache = Array<Void>(repeating: (), count: cell.depth).map {
			Cache(context: context, count: count)
		}
		cache.forEach {
			$0.reset(commandBuffer: commandBuffer)
		}
		super.setup(commandBuffer: commandBuffer, count: count)
	}
}
extension Feedback {
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
	@NSManaged var refer: Int
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, cell: Cell, refer: Int, adapters: (AdapterType, AdapterType)) throws -> Feedback {
		let count: Int = cell.width * cell.width
		let feedback: Feedback = try make()
		feedback.cell = cell
		feedback.refer = refer
		feedback.locationType = adapters.0.rawValue
		feedback.location = Data(count: count * MemoryLayout<Float>.size)
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
		feedback.scaleType = adapters.1.rawValue
		feedback.scale = Data(count: count * MemoryLayout<Float>.size)
		feedback.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		feedback.setup(commandBuffer: commandBuffer, count: count)
		return feedback
	}
}
