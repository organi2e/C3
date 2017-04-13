
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
					distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), f: χ, g: g(0), φ: φ(0), count: width, corrector: corrector)
				case .Identity:
					distributor.activate(commandBuffer: commandBuffer, Δφ: Δ(0), v: χ, g: g(0), φ: φ(0), count: width, corrector: corrector)
				}
				bias.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
				loop.forEach {
					$0.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
				}
				decay?.correct(commandBuffer: commandBuffer, Δφ: Δ(0), φ: φ(0))
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
			state = newValue.isEmpty ? nil :
				context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModePrivate)
		}
	}
	public var target: Array<Float> {
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
			study = newValue.isEmpty ? nil :
				context.make(array: newValue + Array<Float>(repeating: 0, count: max(0, width - newValue.count)), options: .storageModePrivate)
		}
	}
}
extension Cell {
	internal func setup(commandBuffer: CommandBuffer) {
		do {
			let length: Int = width * MemoryLayout<Float>.size
			let ref: Array<Void> = Array<Void>(repeating: (), count: depth)
			fs = ref.map { context.make(length: length, options: .storageModePrivate) }
			vu = ref.map { context.make(length: length, options: .storageModePrivate) }
			vs = ref.map { context.make(length: length, options: .storageModePrivate) }
			gu = ref.map { context.make(length: length, options: .storageModePrivate) }
			gs = ref.map { context.make(length: length, options: .storageModePrivate) }
			du = ref.map { context.make(length: length, options: .storageModePrivate) }
			ds = ref.map { context.make(length: length, options: .storageModePrivate) }
		}
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			(fs+vu+vs+gu+gs+du+ds).forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
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
	func χ(_ offset: Int) -> Buffer {
		assert( fs.count == depth )
		return fs[((offset+fs.count)%fs.count)%fs.count]
	}
	func φ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( vu.count == depth )
		assert( vs.count == depth )
		return (μ: vu[((custom+offset)%vu.count+du.count)%vs.count],
		        σ: vs[((custom+offset)%vs.count+ds.count)%vu.count])
	}
	func g(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( gu.count == depth )
		assert( gs.count == depth )
		return (μ: gu[((custom+offset)%gu.count+gu.count)%gs.count],
		        σ: gs[((custom+offset)%gs.count+gs.count)%gu.count])
		
	}
	func Δ(_ offset: Int) -> (μ: Buffer, σ: Buffer) {
		assert( du.count == depth )
		assert( ds.count == depth )
		return (μ: du[((custom+offset)%du.count+du.count)%ds.count],
		        σ: ds[((custom+offset)%ds.count+ds.count)%du.count])
		
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
	@NSManaged var fs: Array<Buffer>
	@NSManaged var vu: Array<Buffer>
	@NSManaged var vs: Array<Buffer>
	@NSManaged var gu: Array<Buffer>
	@NSManaged var gs: Array<Buffer>
	@NSManaged var du: Array<Buffer>
	@NSManaged var ds: Array<Buffer>
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
		cell.output = Set<Edge>(try output.map{try make(commandBuffer: commandBuffer, output: $0, input: cell, adapters: adapters)})
		cell.input = Set<Edge>(try input.map{try make(commandBuffer: commandBuffer, output: cell, input: $0, adapters: adapters)})
		cell.bias = try make(commandBuffer: commandBuffer, cell: cell, adapters: adapters)
		cell.loop = Set<Feedback>(try recurrent.map{try make(commandBuffer: commandBuffer, cell: cell, refer: $0, adapters: adapters)})
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
