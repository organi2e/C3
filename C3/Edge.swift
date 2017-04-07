//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import CoreData
import Distributor
internal class Edge: Arcane {
	struct Cache {
		let jx: (μ: Buffer, σ: Buffer)
		let ja: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int, encoder: BlitCommandEncoder) {
			jx = (
				μ: context.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate),
				σ: context.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate)
			)
			ja = (
				μ: context.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate),
				σ: context.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate)
			)
			[jx.μ, jx.σ, ja.μ, ja.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
		}
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
}
extension Edge {
	func collect_refresh() {
		input.collect_refresh()
	}
	func collect(collector: Collector, ignore: Set<Cell>) {
		let count: Int = input.width
		access(commandBuffer: collector.order) {
			collector.collect(w: $0, x: input.collect(ignore: ignore), count: count)
		}
	}
}
extension Edge {
	func correct_refresh() {
		output.correct_refresh()
		cache.rotate()
	}
	func correct(corrector: Corrector, ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let (Δφ, φ): (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) = output.correct(ignore: ignore)
		if let x: Buffer = input.state {
			output.distributor.derivate(commandBuffer: corrector.order, Δx: corrector.Δ, j: cache[0].jx, Δφ: Δφ, φ: φ, count: count) {(jacobian: Jacobian)in
				access(commandBuffer: corrector.order) {
					jacobian.jacobian(x: x, a: $0)
				}
				output.jacobian(jacobian: jacobian, j: cache[-1].jx)
			}
			update(commandBuffer: corrector.order) {
				output.distributor.derivate(commandBuffer: corrector.order, Δθ: $0, j: cache[0].ja, Δφ: Δφ, φ: φ, count: count) {(jacobian: Jacobian)in
					access(commandBuffer: corrector.order) {
						jacobian.jacobian(a: $0, x: x)
					}
					output.jacobian(jacobian: jacobian, j: cache[-1].ja)
				}
			}
		}
	}
}
extension Edge {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		let ref: Array<Void> = Array<Void>(repeating: (), count: output.depth)
			cache = RingBuffer<Cache>(buffer: ref.map{
				Cache(context: context, count: count, encoder: encoder)
		}, offset: 0)
		encoder.endEncoding()
	}
	override func awakeFromFetch() {
		super.awakeFromFetch()
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: output.width * input.width)
		commandBuffer.commit()
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer, count: output.width * input.width)
		commandBuffer.commit()
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, output: Cell, input: Cell) throws -> Edge {
		let count: Int = output.width * input.width
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.location = Data(count: count * MemoryLayout<Float>.size)
		edge.scale = Data(count: count * MemoryLayout<Float>.size)
		edge.location.withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) -> Void in
			assert( MemoryLayout<Float>.size == 4 )
			assert( MemoryLayout<UInt32>.size == 4 )
			arc4random_buf(ref, edge.location.count)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
			cblas_sscal(Int32(count/2), 2*Float.pi, ref.advanced(by: count/2), 1)
			vvlogf(ref, ref, [Int32(count/2)])
			cblas_sscal(Int32(count/2), -2, ref, 1)
			vvsqrtf(ref, ref, [Int32(count/2)])
			vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: count/2), 2, vDSP_Length(count/4))
			vDSP_rect(ref, 2, ref, 2, vDSP_Length(count/2))
			//cblas_sscal(Int32(count), 1/Float(input.width), ref, 1)
		}
		edge.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		/*
		edge.scale.withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) -> Void in
			//assert( MemoryLayout<Float>.size == 4 )
			//assert( MemoryLayout<UInt32>.size == 4 )
			//arc4random_buf(ref, edge.location.count)
			/*
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
			cblas_sscal(Int32(count/2), 2*Float.pi, ref.advanced(by: count/2), 1)
			vvlogf(ref, ref, [Int32(count/2)])
			cblas_sscal(Int32(count/2), -2, ref, 1)
			vvsqrtf(ref, ref, [Int32(count/2)])
			vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: count/2), 2, vDSP_Length(count/4))
			vDSP_rect(ref, 2, ref, 2, vDSP_Length(count/2))
			vvexpf(ref, ref, [Int32(count)])
			cblas_sscal(Int32(count), 1/Float(input.width), ref, 1)
			*/
			/*
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
			vvlogf(ref, ref, [Int32(count)])
			cblas_sscal(Int32(count), -sqrt(2), ref, 1)
			vvsqrtf(ref, ref, [Int32(count)])
			*/
			//cblas_sscal(Int32(count), 1/Float(input.width), ref, 1)
		}
		*/
		edge.setup(commandBuffer: commandBuffer, count: count)
		return edge
	}
}
