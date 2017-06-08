//
//  Context.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Accelerate
import CoreData
import Metal

import Adapter
import Distributor
import Optimizer

public enum DistributorType: String {
	case Degenerate = "Degenerate"
	case Gauss = "Gauss"
}
public enum ActivatorType: String {
	case Binary = "Binary"
	case Identity = "Identity"
}
public enum AdapterType: String {
	case Discard = "Discard"
	case Tanh = "Tanh"
	case Linear = "Linear"
	case Floor = "Floor"
	case Regular = "Regular"
	case Logistic = "Logistic"
	case Softplus = "Softplus"
	case Positive = "Positive"
	case RegFloor = "RegFloor"
	case Exponential = "Exponential"
}
public enum OptimizerType {
	case SGD(L2: Float, L1: Float, η: Float)
	case AdaDelta(L2: Float, L1: Float, ρ: Float, ε: Float)
	case Adam(L2: Float, L1: Float, α: Float, β: Float, γ: Float, ε: Float)
	case Adamax(L2: Float, L1: Float, α: Float, β: Float, γ: Float, ε: Float)
	case SMORMS3(L2: Float, L1: Float, α: Float, ε: Float)
}
public class Context: NSManagedObjectContext {
	let mtl: MTLCommandQueue
	let optimizerFactory: (Int) -> Optimizer
	let adapter: Dictionary<AdapterType, (Int)->Adapter>
	let distributor: Dictionary<DistributorType, Distributor>
	enum ErrorCases: Error, CustomStringConvertible {
		case InvalidContext
		case InvalidEntity(name: String)
		case InvalidParameter(key: String, value: Any)
		case NoModelFound
		case NoDeviceFound
		var description: String {
			switch self {
			case .InvalidEntity(let name):
				return "Invalid entity\(name)"
			case .InvalidContext:
				return "This context is invalid"
			case .InvalidParameter(let key, let value):
				return "Invalid value \(value) for \(key)"
			case .NoModelFound:
				return "No CoreData definition was found"
			case .NoDeviceFound:
				return "No available Metal device found"
			}
		}
	}
	public required init?(coder aDecoder: NSCoder) {
		assertionFailure("init(coder:) has not been implemented")
		return nil
	}
	public init(queue: MTLCommandQueue,
	            storage: URL? = nil,
	            optimizer: OptimizerType = .SGD(L2: 0, L1: 0, η: 0),
	            concurrencyType: NSManagedObjectContextConcurrencyType = .privateQueueConcurrencyType) throws {
		let device: Device = queue.device
		mtl = queue
		adapter = try Dictionary<AdapterType, (Int)->Adapter>(dictionaryLiteral:
			(.Discard, Discard.init),
			(.Linear, Linear.init),
			(.Tanh, Tanh.adapter(device: device)),
			(.Floor, Floor.adapter(device: device)),
			(.Regular, Regular.adapter(device: device)),
			(.Positive, Positive.adapter(device: device)),
			(.Softplus, Softplus.adapter(device: device)),
			(.Logistic, Logistic.adapter(device: device)),
			(.RegFloor, RegFloor.adapter(device: device)),
			(.Exponential, Exponential.adapter(device: device))
		)
		distributor = try Dictionary<DistributorType, Distributor>(dictionaryLiteral:
			(.Degenerate, DegenerateDistributor(device: device)),
			(.Gauss, GaussDistributor(device: device))
		)
		switch optimizer {
		case let .SGD(L2, L1, η):
			optimizerFactory = try SGD.optimizer(device: device, L2: L2, L1: L1, η: η)
		case let .AdaDelta(L2, L1, ρ, ε):
			optimizerFactory = try AdaDelta.optimizer(device: device, L2: L2, L1: L1, ρ: ρ, ε: ε)
		case let .Adam(L2, L1, α, β, γ, ε):
			optimizerFactory = try Adam.optimizer(device: device, L2: L2, L1: L1, α: α, β: β, γ: γ, ε: ε)
		case let .Adamax(L2, L1, α, β, γ, ε):
			optimizerFactory = try Adamax.optimizer(device: device, L2: L2, L1: L1, α: α, β: β, γ: γ, ε: ε)
		case let .SMORMS3(L2, L1, α, ε):
			optimizerFactory = try SMORMS3.optimizer(device: device, L2: L2, L1: L1, α: α, ε: ε)
		}
		super.init(concurrencyType: concurrencyType)
		guard let model: NSManagedObjectModel = NSManagedObjectModel.mergedModel(from: [Bundle(for: type(of: self))]) else { throw ErrorCases.NoModelFound }
		let store: NSPersistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		let storetype: String = storage == nil ? NSInMemoryStoreType : ["sqlite", "db"].filter{$0==storage?.pathExtension}.isEmpty ? NSBinaryStoreType : NSSQLiteStoreType
		try store.addPersistentStore(ofType: storetype, configurationName: nil, at: storage, options: nil)
		persistentStoreCoordinator = store
	}
}
extension Context {
	func make(data: Data, options: MTLResourceOptions = []) -> Buffer {
		return data.withUnsafeBytes { mtl.device.makeBuffer(bytes: $0, length: data.count, options: options) }
	}
	func make(length: Int, options: MTLResourceOptions = []) -> Buffer {
		return mtl.device.makeBuffer(length: length, options: options)
	}
	func make<T>(array: Array<T>, options: MTLResourceOptions = []) -> Buffer {
		return mtl.device.makeBuffer(bytes: array, length: array.count * MemoryLayout<T>.size, options: options)
	}
	func make() -> CommandBuffer {
		return mtl.makeCommandBuffer()
	}
	func make(count: Int, type: AdapterType) -> Adapter {
		guard let factory: (Int) -> Adapter = adapter[type] else { fatalError(type.rawValue) }
		return factory(count)
	}
	func make(type: DistributorType) -> Distributor {
		guard let distributor: Distributor = distributor[type] else { fatalError(type.rawValue) }
		return distributor
	}
	func make(count: Int) -> Optimizer {
		return optimizerFactory(count)
	}
}
extension Context {
	public func connect(output: Cell, input: Cell, adapters: (AdapterType, AdapterType)) throws {
		guard output.objectID != input.objectID && output.input.filter({$0.input.objectID==input.objectID}).isEmpty else { return }
		let commandBuffer: CommandBuffer = make()
		try output.input.insert(make(commandBuffer: commandBuffer, output: output, input: input, adapters: adapters))
		commandBuffer.commit()
	}
	public func disconnect(output: Cell, input: Cell) {
		output.input.filter{$0.input.objectID == input.objectID}.forEach(delete)
	}
}
extension Context {
	func make<T: Ground>() throws -> T {
		let name: String = String(describing: T.self)
		guard let entity: T = NSEntityDescription.insertNewObject(forEntityName: name, into: self) as? T else {
			throw ErrorCases.InvalidEntity(name: name)
		}
		return entity
	}
	
	/*
	func count<T: Ground>(predicate: NSPredicate) throws -> Int {
		let name: String = String(describing: T.self)
		let request: NSFetchRequest<T> = NSFetchRequest<T>(entityName: name)
		request.predicate = predicate
		return try count(for: request)
	}
	*/
	/*
	func fetch<T: Ground>(predicate: NSPredicate) throws -> [T] {
		let name: String = String(describing: T.self)
		var cache: [T] = []
		var e: Error?
		func block() {
			let request: NSFetchRequest<T> = NSFetchRequest<T>(entityName: name)
			request.predicate = predicate
			request.returnsObjectsAsFaults = false
			do {
				cache = try fetch(request)
			} catch {
				e = error
			}
		}
		performAndWait(block)
		if let error: Error = e {
			throw error
		}
		return cache
	}
	func remove(object: Ground) {
		func block() {
			delete(object)
		}
		perform(block)
	}
	*/
	public override func save() throws {
		var encounter: Error?
		func done(_: CommandBuffer) {
			do {
				try super.save()
			} catch {
				encounter = error
			}
		}
		let commandBuffer: CommandBuffer = make()
		commandBuffer.addCompletedHandler(done)
		commandBuffer.label = "save"
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
		if let encounter: Error = encounter {
			throw encounter
		}
	}
}
internal typealias Device = MTLDevice
internal typealias Buffer = MTLBuffer
internal typealias CommandQueue = MTLCommandQueue
internal typealias CommandBuffer = MTLCommandBuffer
internal typealias ManagedObjectContext = NSManagedObjectContext
internal typealias BlitCommandEncoder = MTLBlitCommandEncoder
internal typealias ComputeCommandEncoder = MTLComputeCommandEncoder
