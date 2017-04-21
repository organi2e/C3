
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
		rotate()
		state = false
	}
	public func collect() {
		let _: Buffer = collect(visit: [])
	}
	internal func collect(visit: Set<Cell>) -> Buffer {
		guard !visit.contains(self) else {
			return χ(-1)
		}
		if !state {
			func collector(collector: Collector) {
				input.forEach {
					$0.collect(collector: collector, visit: visit.union([self]))
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
			state = true
		}
		return χ(0)
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
		study = false
		delta = false
	}
	public func correct(fix: Set<Cell> = []) {
		let _: (μ: Buffer, σ: Buffer) = correct(fix: [self], visit: [])
	}
	internal func correct(fix: Set<Cell>, visit: Set<Cell>) -> (μ: Buffer, σ: Buffer) {
		guard !visit.contains(self) else {
			return Δ(-1)
		}
		if !delta {
			func corrector(corrector: Corrector) {
				output.forEach {
					$0.correct(corrector: corrector, fix: fix, visit: visit.union([self]))
				}
				if study {
					corrector.correct(χ: χ(0), ϝ: ϝ(0))
				}
			}
			let commandBuffer: CommandBuffer = context.make()
			switch activation {
			case .Binary:
				distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), f: χ(0), g: g(0), φ: φ(0), count: width, corrector: corrector)
			case .Identity:
				distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), v: χ(0), g: g(0), φ: φ(0), count: width, corrector: corrector)
			}
			bias.correct(commandBuffer: commandBuffer, fix: fix, Δφ: Δ(0))
			loop.forEach {
				$0.correct(commandBuffer: commandBuffer, fix: fix, Δφ: Δ(0))
			}
			decay?.correct(commandBuffer: commandBuffer, fix: fix, Δφ: Δ(0))
			commandBuffer.commit()
			delta = true
		}
		return Δ(0)
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
			guard state else { return Array<Float>() }
			let target: Buffer = context.make(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: χ(0), sourceOffset: 0, to: target, destinationOffset: 0, size: min(χ(0).length, target.length))
			encoder.endEncoding()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer {
				target.setPurgeableState(.empty)
			}
			return Array<Float>(UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(target.contents())),
			                                               count: width)
			)
		}
		set {
			let source: Buffer = context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
			                                  options: .storageModeShared)
			let command: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = command.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0,
			             to: χ(0), destinationOffset: 0, size: min(source.length, χ(0).length))
			encoder.endEncoding()
			command.addCompletedHandler { (_) in
				source.setPurgeableState(.empty)
			}
			command.commit()
			state = !newValue.isEmpty
		}
	}
	public var target: Array<Float> {
		get {
			guard study else { return Array<Float>() }
			let target: Buffer = context.make(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: ϝ(0), sourceOffset: 0, to: target, destinationOffset: 0, size: min(ϝ(0).length, target.length))
			encoder.endEncoding()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer {
				target.setPurgeableState(.empty)
			}
			return Array<Float>(UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(target.contents())),
			                                               count: width)
			)
		}
		set {
			let source: Buffer = context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
			                                  options: .storageModeShared)
			let command: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = command.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0,
			             to: ϝ(0), destinationOffset: 0, size: min(source.length, ϝ(0).length))
			encoder.endEncoding()
			command.addCompletedHandler { (_) in
				source.setPurgeableState(.empty)
			}
			command.commit()
			study = !newValue.isEmpty
		}
	}
}
extension Cell {
	@NSManaged private var cache: Array<Cache>
	@NSManaged private var index: Int
	private class Cache: NSObject {
		let χ: Buffer
		let ϝ: Buffer
		let φ: (μ: Buffer, σ: Buffer)
		let g: (μ: Buffer, σ: Buffer)
		let Δ: (μ: Buffer, σ: Buffer)
		init(context: Context, count: Int) {
			let length: Int = count * MemoryLayout<Float>.size
			let option: MTLResourceOptions = .storageModePrivate
			χ = context.make(length: length, options: option)
			ϝ = context.make(length: length, options: option)
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
			[χ, ϝ, φ.μ, φ.σ, g.μ, g.σ, Δ.μ, Δ.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
	}
	internal func χ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].χ
	}
	internal func ϝ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].ϝ
	}
	internal func φ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].φ
	}
	internal func g(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].g
	}
	internal func Δ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.count
		return cache[((offset+index)%cycle+cycle)%cycle].Δ
	}
	internal func rotate() {
		index = ( index + 1 ) % cache.count
	}
	internal func setup(commandBuffer: CommandBuffer) {
		cache = Array<Void>(repeating: (), count: depth).map {
			Cache(context: context, count: width)
		}
		cache.forEach {
			$0.reset(commandBuffer: commandBuffer)
		}
		index = 0
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
	@NSManaged var state: Bool
	@NSManaged var delta: Bool
	@NSManaged var study: Bool
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
