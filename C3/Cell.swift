//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import CoreData
import Distributor
public class Cell: ManagedObject {
	struct Cache {
		let χ: Buffer
		let φ: (μ: Buffer, σ: Buffer)
		let g: (μ: Buffer, σ: Buffer)
		let Δ: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			χ = context.make(length: length, options: .storageModePrivate)
			φ = (
				μ: context.make(length: length, options: .storageModePrivate),
				σ: context.make(length: length, options: .storageModePrivate)
			)
			g = (
				μ: context.make(length: length, options: .storageModePrivate),
				σ: context.make(length: length, options: .storageModePrivate)
			)
			Δ = (
				μ: context.make(length: length, options: .storageModePrivate),
				σ: context.make(length: length, options: .storageModePrivate)
			)
		}
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
	var state: Buffer?
	var study: Buffer?
	var delta: (μ: Buffer, σ: Buffer)?
}
extension Cell {
	public func collect_refresh() {
		input.forEach {
			$0.collect_refresh()
		}
		bias.collect_refresh()
		state = nil
		cache.rotate()
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
			let commandBuffer: CommandBuffer = context.make()
			distributor.activate(commandBuffer: commandBuffer, χ: cache[0].χ, φ: cache[0].φ, count: width) { (collector: Collector) -> Void in
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
	public func correct_refresh() {
		output.forEach {
			$0.correct_refresh()
		}
		bias.correct_refresh()
		delta = nil
		study?.setPurgeableState(.empty)
	}
	public func correct() {
		let _: (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) = correct(ignore: [])
	}
	internal func correct(ignore: Set<Cell>) -> (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		if ignore.contains(self) {
			return (Δφ: cache[-1].Δ, φ: cache[-1].φ)
		} else if let Δφ: (μ: Buffer, σ: Buffer) = delta {
			return (Δφ: Δφ, φ: cache[0].φ)
		} else {
			let commandBuffer: CommandBuffer = context.make()
			distributor.activate(commandBuffer: commandBuffer, Δφ: cache[0].Δ, g: cache[0].g, φ: cache[0].φ, count: width) { (corrector: Corrector) -> Void in
				if let χ: Buffer = state, let ϝ: Buffer = study {
					corrector.correct(χ: χ, ϝ: ϝ)
				} else {
					output.forEach {
						$0.correct(corrector: corrector, ignore: ignore.union([self]))
					}	
				}
			}
			bias.correct(commandBuffer: commandBuffer, Δφ: cache[0].Δ, φ: cache[0].φ)
			commandBuffer.commit()
			delta = cache[0].Δ
			return (Δφ: cache[0].Δ, φ: cache[0].φ)
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
			state = newValue.isEmpty ? nil : context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
			                                              options: .storageModePrivate)
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
			study = newValue.isEmpty ? nil : context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
			                                              options: .storageModePrivate)
		}
	}
}
extension Cell {
	internal func setup(commandBuffer: CommandBuffer) {
		cache = RingBuffer(buffer: Array<Void>(repeating: (), count: depth).map {
			Cache(context: context, count: width)
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
	@NSManaged var decay: Decay?
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
		cell.output = Set<Edge>(try output.map{try make(commandBuffer: commandBuffer, output: $0, input: cell)})
		cell.input = Set<Edge>(try input.map{try make(commandBuffer: commandBuffer, output: cell, input: $0)})
		if recurrent {
			
		}
		cell.bias = try make(commandBuffer: commandBuffer, cell: cell)
		cell.decay = !decay ? nil : try make(commandBuffer: commandBuffer, cell: cell)
		cell.setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
		return cell
	}
	public func fetch(label: String? = nil, width: Int? = nil) throws -> [Cell] {
		func bind(format: String, value: Any?) -> [(String, Any)] {
			guard let value: Any = value else { return [] }
			return [(format, value)]
		}
		let formats: [(String, Any)] = bind(format: "label = %@", value: label) + bind(format: "width = %@", value: width)
		let predicate: NSPredicate = NSPredicate(
			format: formats.map{$0.0}.joined(separator: " and "),
			argumentArray: formats.map{$0.1}
		)
		return try fetch(predicate: predicate)
	}
}
