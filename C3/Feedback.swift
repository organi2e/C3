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
	func collect(commandBuffer: CommandBuffer, collector: Collector) {
		access {
			collector.collect(w: $0, x: cell.χ(refer), count: cell.width)
		}
	}
}
extension Feedback {
	func correct_refresh(commandBuffer: CommandBuffer) {
		rotate()
	}
	func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>, Δφ: (μ: MTLBuffer, σ: MTLBuffer)) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: cell.width)
		if !ignore.contains(cell) {
			change(commandBuffer: commandBuffer) {
				cell.distributor.gradient(commandBuffer: commandBuffer, Δθ: $0, j: j(0), Δφ: Δφ, φ: cell.φ(0), count: count) { connector in
					access {
						connector.connect(a: $0, x: cell.χ(refer))
					}
					cell.connect(connector: connector, feed: j)
				}
			}
		}
	}
}
extension Feedback {
	func connect(connector: Connector, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		access {
			connector.connect(b: $0, y: cell.χ(refer), g: cell.g(refer), j: feed(refer))
		}
	}
}
extension Feedback {
	@NSManaged private var cache: Cache
	private class Cache: NSObject {
		var index: Int
		let array: Array<(μ: Buffer, σ: Buffer)>
		init(context: Context, depth: Int, width: Int) {
			let length: Int = width * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			array = Array<Void>(repeating: (), count: depth)
				.map{
					(μ: context.make(length: length, options: option),
					 σ: context.make(length: length, options: option))
			}
			index = 0
			super.init()
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			array
				.map{[$0.μ, $0.σ]}
				.reduce([], +)
				.forEach{encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)}
			encoder.label = #function
			encoder.endEncoding()
			commandBuffer.label = #function
			commandBuffer.commit()
		}
	}
	internal func rotate() {
		cache.index = ( cache.index + 1 ) % cache.array.count
	}
	internal func j(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle]
	}
	override internal func setup(context: Context, count: Int) {
		cache = Cache(context: context, depth: cell.depth, width: count)
		super.setup(context: context, count: count)
	}
}
extension Feedback {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		try?eval {
			setup(context: $0, count: cell.width * cell.width)
		}
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		try?eval {
			setup(context: $0, count: cell.width * cell.width)
		}
	}
}
extension Feedback {
	@NSManaged var cell: Cell
	@NSManaged var refer: Int
}
extension Context {
	internal func make(cell: Cell, refer: Int, adapters: (AdapterType, AdapterType)) throws -> Feedback {
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
			vDSP_vfill([Float(1)], $0, 1, vDSP_Length(count))
		}
		feedback.setup(context: self, count: count)
		return feedback
	}
}
