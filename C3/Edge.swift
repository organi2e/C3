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
	
}
extension Edge {
	func collect_refresh(commandBuffer: CommandBuffer) {
		input.collect_refresh(commandBuffer: commandBuffer)
		fixing(commandBuffer: commandBuffer)
	}
	func collect(collector: Collector, ignore: Set<Cell>) {
		let count: Int = input.width
		access {
			collector.collect(w: $0, x: input.collect(ignore: ignore), count: count)
		}
	}
}
extension Edge {
	func correct_refresh() {
		output.correct_refresh()
		custom = ( custom + 1 ) % output.depth
	}
	func correct(corrector: Corrector, state: Buffer, ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let (Δφ, φ): (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) = output.correct(ignore: ignore)
		change(commandBuffer: corrector.order) {
			output.distributor.derivate(commandBuffer: corrector.order, Δθ: $0, j: ja(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
				access {
					jacobian.jacobian(a: $0, x: state)
				}
				output.jacobian(jacobian: jacobian, feed: ja)
			}
		}
		output.distributor.derivate(commandBuffer: corrector.order, Δx: corrector.Δ, j: jx(0), Δφ: Δφ, φ: φ, count: count) { jacobian in
			access {
				jacobian.jacobian(x: state, a: $0)
			}
			output.jacobian(jacobian: jacobian, feed: jx)
		}
	}
}
extension Edge {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		do {
			let length: Int = count * MemoryLayout<Float>.size
			let ref: Array<Void> = Array<Void>(repeating: (), count: output.depth)
			jau = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
			jas = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
			jxu = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
			jxs = ref.map {
				context.make(length: length, options: .storageModePrivate)
			}
		}
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			(jau+jau+jxu+jxs).forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
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
	@NSManaged var jxu: Array<Buffer>
	@NSManaged var jxs: Array<Buffer>
	@NSManaged var jau: Array<Buffer>
	@NSManaged var jas: Array<Buffer>
	func ja(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( jau.count == output.depth )
		assert( jas.count == output.depth )
		return (μ: jau[((offset+custom)%jau.count+jau.count)%jau.count],
		        σ: jas[((offset+custom)%jas.count+jas.count)%jas.count])
	}
	func jx(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( jxu.count == output.depth )
		assert( jxs.count == output.depth )
		return (μ: jxu[((offset+custom)%jxu.count+jxu.count)%jxu.count],
		        σ: jxs[((offset+custom)%jxs.count+jxs.count)%jxs.count])
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, output: Cell, input: Cell, adapters: (AdapterType, AdapterType)) throws -> Edge {
		let count: Int = output.width * input.width
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.locationType = adapters.0.rawValue
		edge.location = Data(count: count * MemoryLayout<Float>.size)
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
		}
		edge.scaleType = adapters.1.rawValue
		edge.scale = Data(count: count * MemoryLayout<Float>.size)
		edge.scale.withUnsafeMutableBytes {
			vDSP_vfill([1.0], $0, 1, vDSP_Length(count))
		}
		edge.setup(commandBuffer: commandBuffer, count: count)
		return edge
	}
}
