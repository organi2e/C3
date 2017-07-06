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
import Normalizer

import simd

internal class Edge: Arcane {
	
}
extension Edge {
	func collect_refresh(commandBuffer: CommandBuffer) {
		input.collect_refresh(commandBuffer: commandBuffer)
		fixing(commandBuffer: commandBuffer)
	}
	func collect(commandBuffer: CommandBuffer, collector: Collector, visit: Set<Cell>) {
		let x: Buffer = input.collect(commandBuffer: commandBuffer, visit: visit)
		access {
			collector.collect(w: $0, x: x, count: input.width)
		}
	}
}
extension Edge {
	func correct_refresh(commandBuffer: CommandBuffer) {
		output.correct_refresh(commandBuffer: commandBuffer)
		rotate()
	}
	func correct(commandBuffer: CommandBuffer, corrector: Corrector, ignore: Set<Cell>, visit: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let Δφ: (μ: Buffer, σ: Buffer) = output.correct(commandBuffer: commandBuffer, ignore: ignore, visit: visit)
		let φ: (μ: Buffer, σ: Buffer) = output.φ(0)
		if !ignore.contains(output) {
			change(commandBuffer: corrector.order) {
				output.distributor.gradient(commandBuffer: corrector.order, Δθ: $0, j: ja(0), Δφ: Δφ, φ: φ, count: count) { connector in
					access {
						connector.connect(a: $0, x: input.χ(0))
					}
					output.connect(connector: connector, feed: ja)
				}
			}
		}
		output.distributor.gradient(commandBuffer: corrector.order, Δx: corrector.Δ, j: jx(0), Δφ: Δφ, φ: φ, count: count) { connector in
			access {
				connector.connect(x: input.χ(0), a: $0)
			}
			output.connect(connector: connector, feed: jx)
		}
	}
}
extension Edge {
	@NSManaged private var cache: Cache
	private class Cache: NSObject {
		var index: Int
		let array: Array<(ja: (μ: Buffer, σ: Buffer), jx: (μ: Buffer, σ: Buffer))>
		init(context: Context, depth: Int, height: Int, width: Int) {
			let length: Int = height * width * MemoryLayout<Float>.stride
			index = 0
			array = Array<Void>(repeating: (), count: depth)
				.map{(ja: (μ: context.make(length: length, options: .storageModePrivate),
				           σ: context.make(length: length, options: .storageModePrivate)),
				      jx: (μ: context.make(length: length, options: .storageModePrivate),
				           σ: context.make(length: length, options: .storageModePrivate)))
			}
			super.init()
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			array
				.map{[$0.ja.μ, $0.ja.σ, $0.jx.μ, $0.jx.σ]}
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
	internal func ja(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].ja
	}
	internal func jx(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].jx
	}
	override internal func setup(context: Context, count: Int) {
		cache = Cache(context: context, depth: output.depth, height: output.width, width: input.width)
		super.setup(context: context, count: count)
	}
}
extension Edge {
	override func awakeFromFetch() {
		super.awakeFromFetch()
		try?eval {
			setup(context: $0, count: output.width * input.width)
		}
	}
	override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		try?eval {
			setup(context: $0, count: output.width * input.width)
		}
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	internal func make(output: Cell, input: Cell, adapters: (AdapterType, AdapterType)) throws -> Edge {
		let count: Int = output.width * input.width
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.locationType = adapters.0.rawValue
		edge.location = Data(count: count * MemoryLayout<Float>.stride)
		edge.location.normal(μ: 0.0, σ: 1.0)
		edge.scaleType = adapters.1.rawValue
		edge.scale = Data(count: count * MemoryLayout<Float>.stride)
		edge.scale.fill(const: 1.0)
		edge.setup(context: self, count: count)
		return edge
	}
}
