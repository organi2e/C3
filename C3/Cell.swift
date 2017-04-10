//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import CoreData
import Distributor
public class Cell: Ground {
	struct Cache {
		let χ: Buffer
		let φ: (μ: Buffer, σ: Buffer)
		let g: (μ: Buffer, σ: Buffer)
		let Δ: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int, encoder: BlitCommandEncoder) {
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
			[χ, φ.μ, φ.σ, g.μ, g.σ, Δ.μ, Δ.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
		}
	}
	var cache: RingBuffer<Cache> = RingBuffer<Cache>(buffer: [], offset: 0)
	var state: Buffer? = nil
	var study: Buffer? = nil
	var delta: (μ: Buffer, σ: Buffer)? = nil
}
extension Cell {
	public func collect_refresh() {
		let commandBuffer: CommandBuffer = context.make()
		collect_refresh(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
	internal func collect_refresh(commandBuffer: CommandBuffer) {
		input.forEach {
			$0.collect_refresh(commandBuffer: commandBuffer)
		}
		bias.collect_refresh(commandBuffer: commandBuffer)
		loop.forEach {
			$0.collect_refresh(commandBuffer: commandBuffer)
		}
		decay?.collect_refresh(commandBuffer: commandBuffer)
		cache.rotate()
		state = nil
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
			func collector(collector: Collector) {
				input.forEach {
					$0.collect(collector: collector, ignore: ignore.union([self]))
				}
				bias.collect(collector: collector)
				loop.forEach {
					$0.collect(collector: collector)
				}
				decay?.collect(collector: collector)
			}
			let commandBuffer: CommandBuffer = context.make()
			switch activation {
			case .Binary:
				distributor.activate(commandBuffer: commandBuffer, f: cache[0].χ, g: cache[0].g, φ: cache[0].φ, count: width, collector: collector)
			case .Identity:
				distributor.activate(commandBuffer: commandBuffer, v: cache[0].χ, g: cache[0].g, φ: cache[0].φ, count: width, collector: collector)
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
		loop.forEach {
			$0.correct_refresh()
		}
		decay?.correct_refresh()
		delta = nil
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
			if let χ: Buffer = state {
				let commandBuffer: CommandBuffer = context.make()
				func corrector(corrector: Corrector) {
					output.forEach {
						$0.correct(corrector: corrector, state: χ, ignore: ignore.union([self]))
					}
					if let ϝ: Buffer = study {
						corrector.correct(χ: χ, ϝ: ϝ)
					}
				}
				switch activation {
				case .Binary:
					distributor.activate(commandBuffer: commandBuffer, Δφ: cache[0].Δ, f: χ, g: cache[0].g, φ: cache[0].φ, count: width, corrector: corrector)
				case .Identity:
					distributor.activate(commandBuffer: commandBuffer, Δφ: cache[0].Δ, v: χ, g: cache[0].g, φ: cache[0].φ, count: width, corrector: corrector)
				}
				bias.correct(commandBuffer: commandBuffer, Δφ: cache[0].Δ, φ: cache[0].φ)
				loop.forEach {
					$0.correct(commandBuffer: commandBuffer, Δφ: cache[0].Δ, φ: cache[0].φ)
				}
				decay?.correct(commandBuffer: commandBuffer, Δφ: cache[0].Δ, φ: cache[0].φ)
				commandBuffer.commit()
			}
			delta = cache[0].Δ
			return (Δφ: cache[0].Δ, φ: cache[0].φ)
		}
	}
}
extension Cell {
	func jacobian(jacobian: Jacobian, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		loop.forEach {
			$0.jacobian(jacobian: jacobian, feed: feed)
		}
		decay?.jacobian(jacobian: jacobian, feed: feed)
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
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		cache = RingBuffer(buffer: Array<Void>(repeating: (), count: depth).map {
			Cache(context: context, count: width, encoder: encoder)
		}, offset: 0)
		encoder.endEncoding()
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
		return loop.map{-$0.depth}.reduce(2, max)
	}
	var activation: ActivationType {
		return activationType.activationType
	}
	var distribution: DistributionType {
		return distributionType.distributionType
	}
	var distributor: Distributor {
		return context.make(type: distribution)
	}
}
extension Cell {
	@NSManaged var label: String
	@NSManaged var width: Int
	@NSManaged var distributionType: String
	@NSManaged var activationType: String
	@NSManaged var input: Set<Edge>
	@NSManaged var output: Set<Edge>
	@NSManaged var loop: Set<Feedback>
	@NSManaged var bias: Bias
	@NSManaged var decay: Decay?
}
extension Context {
	public func make(label: String,
	                 width: Int,
	                 distribution: DistributionType = .Degenerate,
	                 activation: ActivationType = .Binary,
	                 adapters: (AdapterType, AdapterType) = (.Linear, .Linear),
	                 output: Set<Cell> = Set<Cell>(),
	                 input: Set<Cell> = Set<Cell>(),
	                 decay: Bool = false,
	                 recurrent: Array<Int> = Array<Int>()) throws -> Cell {
		guard 0 < width else { throw ErrorCase.InvalidParameter(key: "width", value: width) }
		guard recurrent.filter({ 0 <= $0 }).isEmpty else { throw ErrorCase.InvalidParameter(key: "recurrent", value: recurrent) }
		let commandBuffer: CommandBuffer = make()
		let cell: Cell = try make()
		cell.label = label
		cell.width = width
		cell.distributionType = distribution.rawValue
		cell.activationType = activation.rawValue
		cell.output = Set<Edge>(try output.map{try make(commandBuffer: commandBuffer, output: $0, input: cell, adapters: adapters)})
		cell.input = Set<Edge>(try input.map{try make(commandBuffer: commandBuffer, output: cell, input: $0, adapters: adapters)})
		cell.bias = try make(commandBuffer: commandBuffer, cell: cell, adapters: adapters)
		cell.loop = Set<Feedback>(try recurrent.map{try make(commandBuffer: commandBuffer, cell: cell, depth: $0, adapters: adapters)})
		cell.decay = !decay ? nil : try make(commandBuffer: commandBuffer, cell: cell)
		cell.setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
		return cell
	}
}
extension Context {
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
private extension String {
	var activationType: ActivationType {
		guard let activationType: ActivationType = ActivationType(rawValue: self) else { fatalError(self) }
		return activationType
	}
	var distributionType: DistributionType {
		guard let distributionType: DistributionType = DistributionType(rawValue: self) else { fatalError(self) }
		return distributionType
	}
	
}
