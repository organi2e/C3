//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Distributor

internal class Edge: Arcane {
	var jx: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: [], offset: 0)
	var ja: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: [], offset: 0)
}
extension Edge {
	func collect_clear(commandBuffer: CommandBuffer) {
		input.collect_clear()
	}
	func collect(collector: Collector, ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let x: Buffer = input.collect(ignore: ignore)
		refresh(commandBuffer: collector.order) {
			
			output.distributor.flush(commandBuffer: collector.order, θ: jx.rotate())
			output.distributor.jacobian(commandBuffer: collector.order, Σ: jx[0], x: x, a: $0, count: count)
			
			output.distributor.flush(commandBuffer: collector.order, θ: ja.rotate())
			output.distributor.jacobian(commandBuffer: collector.order, Σ: ja[0], a: $0, x: x, count: count)
			
			collector.collect(w: $0, x: x, count: count.cols)
			
		}
	}
}
extension Edge {
	func correct_clear(commandBuffer: CommandBuffer) {
		output.correct_clear()
	}
	func correct(corrector: Corrector, ignore: Set<Cell>) {
		
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let (Δφ, φ): (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) = output.correct(ignore: ignore)
		
		output.distributor.jacobian(commandBuffer: corrector.order, j: ja[0], Σ: ja[0], φ: φ, count: count)
		update(commandBuffer: corrector.order) {
			output.distributor.derivate(commandBuffer: corrector.order, Δ: $0, j: ja[0], Δφ: Δφ, count: count)
		}
		
		output.distributor.jacobian(commandBuffer: corrector.order, j: jx[0], Σ: jx[0], φ: φ, count: count)
		corrector.correct(j: jx[0], Δ: Δφ, count: output.width)
		
	}
}
extension Edge {
	override func setup(commandBuffer: CommandBuffer, count: Int) {
		super.setup(commandBuffer: commandBuffer, count: count)
		let ref: Array<Void> = Array<Void>(repeating: (), count: output.depth)
		ja = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: ref.map {(
			μ: context.make(length: count * MemoryLayout<Float>.size),
			σ: context.make(length: count * MemoryLayout<Float>.size)
		)}, offset: 0)
		jx = RingBuffer<(μ: Buffer, σ: Buffer)>(buffer: ref.map {(
			μ: context.make(length: count * MemoryLayout<Float>.size),
			σ: context.make(length: count * MemoryLayout<Float>.size)
		)}, offset: 0)
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
extension Edge {
	var depth: Int { return 2 }
}
extension Context {
	internal func make(commandBuffer: CommandBuffer, output: Cell, input: Cell) throws -> Edge {
		typealias T = Float
		let count: Int = output.width * input.width
		let edge: Edge = try make()
		let μ: T = T(0)
		let σ: T = T(1)
		edge.output = output
		edge.input = input
		edge.location = Data(bytes: Array<T>(repeating: μ, count: count), count: count * MemoryLayout<T>.size)
		edge.logscale = Data(bytes: Array<T>(repeating: σ, count: count), count: count * MemoryLayout<T>.size)
		edge.setup(commandBuffer: commandBuffer, count: count)
		return edge
	}
}
