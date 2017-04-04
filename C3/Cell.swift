//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Distributor

public class Cell: ManagedObject {
	struct Cache {
		let χ: Buffer
		let φ: (μ: Buffer, σ: Buffer)
		let g: (μ: Buffer, σ: Buffer)
		let Δφ: (μ: Buffer, σ: Buffer)
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
	var state: Buffer?
	var study: Buffer?
	var delta: (μ: Buffer, σ: Buffer)?
}
extension Cell {
	public func collect_clear() {
		state = nil
		let commandBuffer: CommandBuffer = context.make()
		input.forEach {
			$0.collect_clear(commandBuffer: commandBuffer)
		}
		commandBuffer.commit()
	}
	public func collect() {
		let _: Buffer = collect(ignore: [])
	}
	internal func collect(ignore: Set<Cell>) -> Buffer {
		if ignore.contains(self) {
			return cache[-1].χ
		} else if let state: Buffer = state {
			return state
		} else {
			let latest: Cache = cache.rotate()
			let commandBuffer: CommandBuffer = context.make()
			context.gaussDistributor.activate(commandBuffer: commandBuffer, χ: latest.χ, φ: latest.φ, count: width) { (collector: Collector) -> Void in
				input.forEach { $0.collect(collector: collector, ignore: ignore.union([self])) }
				bias.collect(collector: collector)
			}
			commandBuffer.commit()
			state = cache[0].χ
			return cache[0].χ
		}
	}
}
extension Cell {
	public func correct_clear() {
		delta = nil
		let commandBuffer: CommandBuffer = context.make()
		output.forEach {
			$0.correct_clear(commandBuffer: commandBuffer)
		}
		commandBuffer.commit()
	}
	public func correct() {
		let _ = correct(ignore: [])
	}
	internal func correct(ignore: Set<Cell>) -> (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		if ignore.contains(self) {
			return (Δφ: cache[-1].Δφ, φ: cache[-1].φ)
		} else if let Δφ: (μ: Buffer, σ: Buffer) = delta {
			return (Δφ: Δφ, φ: cache[0].φ)
		} else {
			let commandBuffer: CommandBuffer = context.make()
			context.gaussDistributor.activate(commandBuffer: commandBuffer, Δφ: cache[0].Δφ, g: cache[0].g, φ: cache[0].φ, count: width) { (corrector: Corrector) -> Void in
				if let ϝ: Buffer = study, let χ: Buffer = state {
					corrector.correct(χ: χ, ϝ: ϝ)
				} else {
					output.forEach {
						$0.correct(corrector: corrector, ignore: ignore.union([self]))
					}
				}
			}
			bias.correct(commandBuffer: commandBuffer, Δφ: cache[0].Δφ, φ: cache[0].φ)
			commandBuffer.commit()
			delta = cache[0].Δφ
			return (Δφ: cache[0].Δφ, φ: cache[0].φ)
		}
	}
}

extension Cell {
	var source: Array<Float> {
		get {
			guard let source: Buffer = state else { return [] }
			let target: Buffer = context.make(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: min(source.length, target.length))
			encoder.endEncoding()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer { target.setPurgeableState(.empty) }
			return Array<Float>(UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(target.contents())), count: width))
		}
		set {
			state = newValue.isEmpty ? nil : context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModePrivate)
		}
	}
	var target: Array<Float> {
		get {
			guard let source: Buffer = study else { return [] }
			let target: Buffer = context.make(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: min(source.length, target.length))
			encoder.endEncoding()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer { target.setPurgeableState(.empty) }
			return Array<Float>(UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(target.contents())), count: width))
		}
		set {
			study = newValue.isEmpty ? nil : context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModePrivate)
		}
	}
}
extension Cell {
	internal func setup(commandBuffer: CommandBuffer) {
		let length: Int = width * MemoryLayout<Float>.size
		let options: MTLResourceOptions = .storageModePrivate
		cache = RingBuffer(buffer: Array<Void>(repeating: (), count: depth).map {
			Cache(χ: context.make(length: length, options: options),
			      φ: (μ: context.make(length: length, options: options), σ: context.make(length: length, options: options)),
			      g: (μ: context.make(length: length, options: options), σ: context.make(length: length, options: options)),
			      Δφ: (μ: context.make(length: length, options: options), σ: context.make(length: length, options: options)))
		}, offset: 0)
	}
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
	public override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		let commandBuffer: CommandBuffer = context.make()
		setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
}
extension Cell {
	var depth: Int {
		return 2
	}
	var distributor: Distributor {
		return context.gaussDistributor
	}
}
extension Cell {
	@NSManaged var label: String
	@NSManaged var width: Int
	@NSManaged var input: Set<Edge>
	@NSManaged var output: Set<Edge>
	@NSManaged var bias: Bias
}
extension Context {
	public func make(label: String,
	                 width: Int,
	                 output: [Cell] = [],
	                 input: [Cell] = [],
	                 decay: Bool = false,
	                 recurrent: Bool = false) throws -> Cell {
		assert(0<width)
		let commandBuffer: CommandBuffer = make()
		let cell: Cell = try make()
		cell.label = label
		cell.width = width
		try output.forEach {
			let _: Edge = try make(commandBuffer: commandBuffer, output: $0, input: cell)
		}
		try input.forEach {
			let _: Edge = try make(commandBuffer: commandBuffer, output: cell, input: $0)
		}
		if recurrent {
			
		}
		cell.bias = try make(commandBuffer: commandBuffer, cell: cell)
		if decay {
			
		}
		cell.setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
		return cell
	}
	public func fetch(label: String? = nil, width: Int? = nil) throws -> [Cell] {
		func bind(format: String, value: Any?) -> [(String, Any)] {
			guard let value: Any = value else { return [] }
			return [(format, value)]
		}
		let request: NSFetchRequest<Cell> = NSFetchRequest<Cell>(entityName: String(describing: Cell.self))
		let formats: [(String, Any)] = bind(format: "label = %@", value: label) + bind(format: "width = %@", value: width)
		request.predicate = formats.isEmpty ? nil : NSPredicate(
			format: formats.map { $0.0 }.joined(separator: " and "),
			argumentArray: formats.map { $0.1 }
		)
		return try fetch(request)
	}
}
