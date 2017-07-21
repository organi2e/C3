
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
import simd
public class Cell: Ground {
	
}
extension Cell {
	public func collect_refresh() throws {
		try eval {
			let commandBuffer: CommandBuffer = $0.make()
			collect_refresh(commandBuffer: commandBuffer)
			commandBuffer.label = #function
			commandBuffer.commit()
		}
	}
	internal func collect_refresh(commandBuffer: CommandBuffer) {
		if state {
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
	}
	public func collect() throws {
		return try eval {
			let commandBuffer: CommandBuffer = $0.make()
			let _: Buffer = collect(commandBuffer: commandBuffer, visit: Set<Cell>())
			commandBuffer.label = #function
			commandBuffer.commit()
		}
	}
	internal func collect(commandBuffer: CommandBuffer, visit: Set<Cell>) -> Buffer {
		guard !visit.contains(self) else {
			return χ(-1)
		}
		if !state {
			func collect(collector: Collector) {
				input.forEach {
					$0.collect(commandBuffer: commandBuffer, collector: collector, visit: visit.union([self]))
				}
				bias.collect(commandBuffer: commandBuffer, collector: collector)
				loop.forEach {
					$0.collect(commandBuffer: commandBuffer, collector: collector)
				}
				decay?.collect(commandBuffer: commandBuffer, collector: collector)
			}
			distributor.activate(commandBuffer: commandBuffer, φ: φ(0), count: width, collect: collect)
			switch activation {
			case .Binary:
				distributor.activate(commandBuffer: commandBuffer, f: χ(0), g: g(0), φ: φ(0), count: width)
			case .Identity:
				distributor.activate(commandBuffer: commandBuffer, v: χ(0), g: g(0), φ: φ(0), count: width)
			}
			state = true
		}
		return χ(0)
	}
}
extension Cell {
	public func correct_refresh() throws {
		try eval {
			let commandBuffer: CommandBuffer = $0.make()
			correct_refresh(commandBuffer: commandBuffer)
			commandBuffer.label = #function
			commandBuffer.commit()
		}
	}
	internal func correct_refresh(commandBuffer: CommandBuffer) {
		if delta {
			output.forEach {
				$0.correct_refresh(commandBuffer: commandBuffer)
			}
			bias.correct_refresh(commandBuffer: commandBuffer)
			loop.forEach {
				$0.correct_refresh(commandBuffer: commandBuffer)
			}
			decay?.correct_refresh(commandBuffer: commandBuffer)
			delta = false
		}
		if study {
			study = false
		}
	}
	public func correct(ignore: Set<Cell> = Set<Cell>()) throws {
		return try eval {
			let commandBuffer: CommandBuffer = $0.make()
			let _: (μ: Buffer, σ: Buffer) = correct(commandBuffer: commandBuffer, ignore: ignore.union([self]), visit: [])
			commandBuffer.label = #function
			commandBuffer.commit()
		}
	}
	internal func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>, visit: Set<Cell>) -> (μ: Buffer, σ: Buffer) {
		guard !visit.contains(self) else {
			return Δφ(-1)
		}
		if !delta {
			switch activation {
			case .Binary:
				func corrector(corrector: Corrector) {
					output.forEach {
						$0.correct(commandBuffer: commandBuffer, corrector: corrector, ignore: ignore, visit: visit.union([self]))
					}
					if study {
//						corrector.correct(χ: χ(0), ϝ: ϝ(0))
						corrector.correct(φ: φ(0), f: ϝ(0))
					}
				}
				distributor.derivate(commandBuffer: commandBuffer, Δ: Δ(0), count: width, correct: corrector)
				distributor.derivate(commandBuffer: commandBuffer, Δφ: Δφ(0), Δ: Δ(0), f: χ(0), g: g(0), φ: φ(0), count: width)
			case .Identity:
				func corrector(corrector: Corrector) {
					output.forEach {
						$0.correct(commandBuffer: commandBuffer, corrector: corrector, ignore: ignore, visit: visit.union([self]))
					}
					if study {
//						corrector.correct(χ: χ(0), ϝ: ϝ(0))
						corrector.correct(φ: φ(0), v: ϝ(0))
					}
				}
				distributor.derivate(commandBuffer: commandBuffer, Δ: Δ(0), count: width, correct: corrector)
				distributor.derivate(commandBuffer: commandBuffer, Δφ: Δφ(0), Δ: Δ(0), v: χ(0), g: g(0), φ: φ(0), count: width)
			}
			if 0 < regularizer {
				distributor.derivate(commandBuffer: commandBuffer, Δφ: Δφ(0), θ: θ, φ: φ(0), γ: regularizer, count: width)
			}
			bias.correct(commandBuffer: commandBuffer, ignore: ignore, Δφ: Δφ(0))
			loop.forEach {
				$0.correct(commandBuffer: commandBuffer, ignore: ignore, Δφ: Δφ(0))
			}
			decay?.correct(commandBuffer: commandBuffer, ignore: ignore, Δφ: Δφ(0))
			delta = true
		}
		return Δφ(0)
	}
}
extension Cell {
	func connect(connector: Connector, feed: (Int) -> (μ: Buffer, σ: Buffer)) {
		loop.forEach {
			$0.connect(connector: connector, feed: feed)
		}
		decay?.connect(connector: connector, feed: feed)
	}
}
extension Cell {
	public var expect: Array<Float> {
		return (try?eval {
			let e: Buffer = $0.make(length: width * MemoryLayout<Float>.stride, options: .storageModeShared)
			let commandBuffer: CommandBuffer = $0.make()
			switch activation {
			case .Binary:
				distributor.activate(commandBuffer: commandBuffer, p: e, φ: φ(0), count: width)
			case .Identity:
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				encoder.copy(from: φ(0).μ, sourceOffset: 0, to: e, destinationOffset: 0, size: min(φ(0).μ.length, e.length))
				encoder.label = #function
				encoder.endEncoding()
			}
			commandBuffer.label = #function
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer {
				e.setPurgeableState(.empty)
			}
			return e.array
		}) ?? Array<Float>(repeating: 0, count: width)
	}
}
extension Cell {
	public var source: Array<Float> {
		get {
			guard state else { return Array<Float>() }
			return (try?eval {
				let target: Buffer = $0.make(length: width * MemoryLayout<Float>.stride, options: .storageModeShared)
				let commandBuffer: CommandBuffer = $0.make()
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				encoder.copy(from: χ(0), sourceOffset: 0, to: target, destinationOffset: 0, size: min(χ(0).length, target.length))
				encoder.label = #function
				encoder.endEncoding()
				commandBuffer.label = #function
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				defer {
					target.setPurgeableState(.empty)
				}
				return target.array
			}) ?? Array<Float>(repeating: 0, count: width)
		}
		set {
			state = (try?eval {
				let source: Buffer = $0.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
				                             options: .storageModeShared)
				let commandBuffer: CommandBuffer = $0.make()
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				encoder.copy(from: source, sourceOffset: 0,
				             to: χ(0), destinationOffset: 0, size: min(source.length, χ(0).length))
				encoder.label = #function
				encoder.endEncoding()
				commandBuffer.addCompletedHandler { (_) in
					source.setPurgeableState(.empty)
				}
				commandBuffer.label = #function
				commandBuffer.commit()
				return newValue.isEmpty
			}) == false
		}
	}
	public var target: Array<Float> {
		get {
			guard study else { return Array<Float>() }
			return (try?eval {
				let target: Buffer = $0.make(length: width * MemoryLayout<Float>.stride, options: .storageModeShared)
				let commandBuffer: CommandBuffer = $0.make()
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				encoder.copy(from: ϝ(0), sourceOffset: 0, to: target, destinationOffset: 0, size: min(ϝ(0).length, target.length))
				encoder.label = #function
				encoder.endEncoding()
				commandBuffer.label = #function
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				defer {
					target.setPurgeableState(.empty)
				}
				return target.array
			}) ?? Array<Float>(repeating: 0, count: width)
		}
		set {
			study = (try?eval {
				let source: Buffer = $0.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)),
				                             options: .storageModeShared)
				let commandBuffer: CommandBuffer = $0.make()
				let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
				encoder.copy(from: source, sourceOffset: 0,
				             to: ϝ(0), destinationOffset: 0, size: min(source.length, ϝ(0).length))
				encoder.label = #function
				encoder.endEncoding()
				commandBuffer.addCompletedHandler { (_) in
					source.setPurgeableState(.empty)
				}
				commandBuffer.label = #function
				commandBuffer.commit()
				return newValue.isEmpty
			}) == false
		}
	}
}
extension Cell {
	@NSManaged private var cache: Cache
	private class Cache: NSObject {
		var index: Int
		let array: Array<(χ: Buffer, ϝ: Buffer, Δ: Buffer, φ: (μ: Buffer, σ: Buffer), g: (μ: Buffer, σ: Buffer), Δφ: (μ: Buffer, σ: Buffer))>
		let theta: (μ: Buffer, g: Buffer)
		let distributor: Distributor
		init(context: Context, distribution: DistributorType, depth: Int, width: Int) {
			let length: Int = width * MemoryLayout<Float>.stride
			let option: MTLResourceOptions = .storageModePrivate
			index = 0
			array = Array<Void>(repeating: (), count: depth)
				.map{(χ: context.make(length: length, options: option),
				      ϝ: context.make(length: length, options: option),
				      Δ: context.make(length: length, options: option),
				      φ: (μ: context.make(length: length, options: option), σ: context.make(length: length, options: option)),
				      g: (μ: context.make(length: length, options: option), σ: context.make(length: length, options: option)),
						 Δφ: (μ: context.make(length: length, options: option), σ: context.make(length: length, options: option)))
			}
			theta = (μ: context.make(length: width * MemoryLayout<float2>.stride, options: option),
			         g: context.make(length: width * MemoryLayout<float4>.stride, options: option))
			distributor = context.make(type: distribution)
			super.init()
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			array
				.map{[$0.g.μ, $0.g.σ, $0.Δφ.μ, $0.Δφ.σ, $0.φ.μ, $0.φ.σ, $0.χ, $0.ϝ]}
				.reduce([theta.μ, theta.g], +)
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
	internal func χ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].χ
	}
	internal func ϝ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].ϝ
	}
	internal func φ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].φ
	}
	internal func g(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].g
	}
	internal func Δ(_ offset: Int) -> Buffer {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].Δ
	}
	internal func Δφ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		let cycle: Int = cache.array.count
		return cache.array[((offset+cache.index)%cycle+cycle)%cycle].Δφ
	}
	internal var θ: (μ: Buffer, g: Buffer) {
		return cache.theta
	}
	internal var distributor: Distributor {
		return cache.distributor
	}
	internal func setup(context: Context) {
		cache = Cache(context: context, distribution: distribution, depth: depth, width: width)
	}
}
extension Cell {
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		try?eval {
			setup(context: $0)
		}
	}
	public override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		try?eval {
			setup(context: $0)
		}
	}
}
extension Cell {
	@NSManaged var activatorType: String
	var activation: ActivatorType {
		return activatorType.activatorType
	}
}
extension Cell {
	@NSManaged var distributorType: String
	var distribution: DistributorType {
		return distributorType.distributorType
	}
}
extension Cell {
	@NSManaged var regularizer: Float
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
	@NSManaged var input: Set<Edge>
	@NSManaged var output: Set<Edge>
	@NSManaged var loop: Set<Feedback>
	@NSManaged var bias: Bias
	@NSManaged var decay: Decay?
}
extension Context {
	public func make(label: String,
	                 width: Int,
	                 distributor: DistributorType = .Degenerate,
	                 regularizer: Float = 0,
	                 activator: ActivatorType = .Binary,
	                 adapters: (AdapterType, AdapterType) = (.Linear, .Softplus),
	                 output: Set<Cell> = Set<Cell>(),
	                 input: Set<Cell> = Set<Cell>(),
	                 decay: Bool = false,
	                 recurrent: Array<Int> = Array<Int>()) throws -> Cell {
		guard 0 < width else { throw ErrorCases.InvalidParameter(key: "width", value: width) }
		guard recurrent.filter({ 0 <= $0 }).isEmpty else { throw ErrorCases.InvalidParameter(key: "recurrent", value: recurrent) }
		let cell: Cell = try make()
		cell.label = label
		cell.width = width
		cell.depth = -recurrent.reduce(-2, min)
		cell.distributorType = distributor.rawValue
		cell.regularizer = regularizer
		cell.activatorType = activator.rawValue
		cell.output = try Set<Edge>(output.map{try make(output: $0, input: cell, adapters: adapters)})
		cell.input = try Set<Edge>(input.map{try make(output: cell, input: $0, adapters: adapters)})
		cell.bias = try make(cell: cell, adapters: adapters)
		cell.loop = try Set<Feedback>(recurrent.map{try make(cell: cell, refer: $0, adapters: adapters)})
		cell.decay = try !decay ? nil : make(cell: cell)
		cell.setup(context: self)
		return cell
	}
}
extension Context {
	private func make<T: NSManagedObject>(label: String? = nil, width: Int? = nil) -> NSFetchRequest<T> {
		let formats: Array<(String, Any)> = Array<(String, Any?)>(arrayLiteral: ("label = %@", label), ("width = %@", width))
			.flatMap {
				guard let value: Any = $1 else {
					return nil
				}
				return Array<(String, Any)>(arrayLiteral: ($0, value))
			}
			.reduce(Array<(String, Any)>(), +)
		let request: NSFetchRequest<T> = NSFetchRequest<T>(entityName: String(describing: T.self))
		request.predicate = NSPredicate(format: formats.map{$0.0}.joined(separator: " and "), argumentArray: formats.map{$0.1})
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
	var activatorType: ActivatorType {
		guard let activationType: ActivatorType = ActivatorType(rawValue: self) else { fatalError(self) }
		return activationType
	}
	var distributorType: DistributorType {
		guard let distributionType: DistributorType = DistributorType(rawValue: self) else { fatalError(self) }
		return distributionType
	}
	
}
