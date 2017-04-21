
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
		custom = ( custom + 1 ) % depth
		state = nil
	}
	public func collect() {
		let _: Buffer = collect(ignore: [])
	}
	internal func collect(ignore: Set<Cell>) -> Buffer {
		if ignore.contains(self) {
			return χ(-1)
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
				distributor.activate(commandBuffer: commandBuffer, f: χ(0), g: g(0), φ: φ(0), count: width, collector: collector)
			case .Identity:
				distributor.activate(commandBuffer: commandBuffer, v: χ(0), g: g(0), φ: φ(0), count: width, collector: collector)
			}
			commandBuffer.commit()
			state = χ(0)
			return χ(0)
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
		study = nil
		gradu = nil
		grads = nil
	}
	public func correct() {
		let _: (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) = correct(ignore: [])
	}
	internal func correct(ignore: Set<Cell>) -> (Δφ: (μ: Buffer, σ: Buffer), φ: (μ: Buffer, σ: Buffer)) {
		if ignore.contains(self) {
			return (Δφ: Δ(-1), φ: φ(-1))
		} else if let μ: Buffer = gradu, let σ: Buffer = grads {
			return (Δφ: (μ: μ, σ: σ), φ: φ(0))
		} else {
			if let state: Buffer = state {
				func corrector(corrector: Corrector) {
					output.forEach {
						$0.correct(corrector: corrector, state: state, ignore: ignore.union([self]))
					}
					if let study: Buffer = study {
						corrector.correct(χ: state, ϝ: study)
					}
				}
				let commandBuffer: CommandBuffer = context.make()
				switch activation {
				case .Binary:
					distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), f: state, g: g(0), φ: φ(0), count: width, corrector: corrector)
				case .Identity:
					distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), v: state, g: g(0), φ: φ(0), count: width, corrector: corrector)
				}
				bias.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
				loop.forEach {
					$0.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
				}
				decay?.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
				commandBuffer.commit()
			} else {
				let commandBuffer: CommandBuffer = context.make()
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				[Δ(0).μ, Δ(0).σ].forEach {
					encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
				}
				encoder.endEncoding()
				commandBuffer.commit()
			}
			gradu = Δ(0).μ
			grads = Δ(0).σ
			return (Δφ: Δ(0), φ: φ(0))
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
	public var source: Array<Float> {
		get {
			guard let source: Buffer = state else { return Array<Float>() }
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
			state = newValue.isEmpty ? nil :
				context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModeShared)
		}
	}
	public var target: Array<Float> {
		get {
			guard let source: Buffer = study else { return Array<Float>() }
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
			study = newValue.isEmpty ? nil :
				context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModeShared)
		}
	}
}
extension Cell {
	@NSManaged private var cache: Array<Cache>
	private class Cache: NSObject {
		let χ: Buffer
		let φ: (μ: Buffer, σ: Buffer)
		let g: (μ: Buffer, σ: Buffer)
		let Δ: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			χ = context.make(length: length, options: option)
			φ = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			g = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			Δ = (
				μ: context.make(length: length, options: option),
				σ: context.make(length: length, options: option)
			)
			super.init()
		}
		func reset(commandBuffer: CommandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[χ, φ.μ, φ.σ, g.μ, g.σ, Δ.μ, Δ.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
	}
	internal func χ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.count
		return cache[((offset+custom)%cycle+cycle)%cycle].χ
	}
	internal func φ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+custom)%cycle+cycle)%cycle].φ
	}
	internal func g(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+custom)%cycle+cycle)%cycle].g
		
	}
	internal func Δ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+custom)%cycle+cycle)%cycle].Δ
	}
	internal func setup(commandBuffer: CommandBuffer) {
		cache = Array<Void>(repeating: (), count: depth).map {
			Cache(context: context, count: width)
		}
		cache.forEach {
			$0.reset(commandBuffer: commandBuffer)
		}
	}
}
extension Cell {
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
	@NSManaged var state: Buffer?
	@NSManaged var study: Buffer?
	@NSManaged var gradu: Buffer?
	@NSManaged var grads: Buffer?
}
extension Cell {
	@NSManaged var label: String
	@NSManaged var width: Int
	@NSManaged var depth: Int
	@NSManaged var distributionType: String
	@NSManaged var activationType: String
	@NSManaged var input: Set<Edge>
	@NSManaged var output: Set<Edge>
	@NSManaged var loop: Set<Feedback>
	@NSManaged var bias: Bias
	@NSManaged var decay: Decay?
	@NSManaged public var pliable: Bool
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
		cell.depth = recurrent.map{-$0}.reduce(2, max)
		cell.distributionType = distribution.rawValue
		cell.activationType = activation.rawValue
		cell.output = try Set<Edge>(output.map{try make(commandBuffer: commandBuffer, output: $0, input: cell, adapters: adapters)})
		cell.input = try Set<Edge>(input.map{try make(commandBuffer: commandBuffer, output: cell, input: $0, adapters: adapters)})
		cell.bias = try make(commandBuffer: commandBuffer, cell: cell, adapters: adapters)
		cell.loop = try Set<Feedback>(recurrent.map{try make(commandBuffer: commandBuffer, cell: cell, refer: $0, adapters: adapters)})
		cell.decay = try !decay ? nil : make(commandBuffer: commandBuffer, cell: cell)
		cell.setup(commandBuffer: commandBuffer)
		commandBuffer.commit()
		return cell
	}
}
extension Context {
	private func make<T: NSManagedObject>(label: String? = nil, width: Int? = nil) -> NSFetchRequest<T> {
		let formats: Array<(String, Any)> = Array<(String, Any?)>(arrayLiteral: ("label = %@", label), ("width = %@", width)).map {
			guard let value: Any = $1 else { return Array<(String, Any)>() }
			return Array<(String, Any)>(arrayLiteral: ($0, value))
		}.reduce(Array<(String, Any)>(), +)
		let request: NSFetchRequest<T> = NSFetchRequest<T>(entityName: String(describing: T.self))
		request.predicate = NSPredicate(format: formats.map{$0.0}.joined(separator: " and "),
		                                argumentArray: formats.map{$0.1}
		)
		return request
	}
	public func count(label: String? = nil, width: Int? = nil) throws -> Int {
		let request: NSFetchRequest<Cell> = make(label: label, width: width)
		return try count(for: request)
	}
	public func fetch(label: String? = nil, width: Int? = nil) throws -> [Cell] {
		let request: NSFetchRequest<Cell> = make(label: label, width: width)
		return try fetch(request)
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
