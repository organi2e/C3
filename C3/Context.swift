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

public enum DistributionType: String {
	case Degenerate = "Degenerate"
	case Gauss = "Gauss"
}
public enum ActivationType: String {
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
public class Context: NSManagedObjectContext {
	let mtl: MTLCommandQueue
	let optimizerFactory: (Int) -> Optimizer
	let adapter: Dictionary<AdapterType, (Int)->Adapter>
	let distributor: Dictionary<DistributionType, Distributor>
	enum ErrorCase: Error, CustomStringConvertible {
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
	public init(queue: MTLCommandQueue,
	            storage: URL? = nil,
	            optimizer: (MTLDevice) throws -> (Int) -> Optimizer = SGD.factory(),
	            concurrencyType: NSManagedObjectContextConcurrencyType = .privateQueueConcurrencyType) throws {
		let device: Device = queue.device
		mtl = queue
		adapter = try {
			var result: Dictionary<AdapterType, (Int)->Adapter> = Dictionary<AdapterType, (Int)->Adapter>()
			result.updateValue(Discard.init, forKey: .Discard)
			result.updateValue(Linear.init, forKey: .Linear)
			result.updateValue(try Tanh.adapter(device: $0), forKey: .Tanh)
			result.updateValue(try Floor.adapter(device: $0), forKey: .Floor)
			result.updateValue(try Regular.adapter(device: $0), forKey: .Regular)
			result.updateValue(try Positive.adapter(device: $0), forKey: .Positive)
			result.updateValue(try Softplus.adapter(device: $0), forKey: .Softplus)
			result.updateValue(try Logistic.adapter(device: $0), forKey: .Logistic)
			result.updateValue(try RegFloor.adapter(device: $0), forKey: .RegFloor)
			result.updateValue(try Exponential.adapter(device: $0), forKey: .Exponential)
			return result
		} (device)
		distributor = try {
			var result: Dictionary<DistributionType, Distributor> = Dictionary<DistributionType, Distributor>()
			result.updateValue(try DegenerateDistributor(device: $0), forKey: .Degenerate)
			result.updateValue(try GaussDistributor(device: $0), forKey: .Gauss)
			return result
		} (device)
		optimizerFactory = try optimizer(device)
		super.init(concurrencyType: concurrencyType)
		guard let model: NSManagedObjectModel = NSManagedObjectModel.mergedModel(from: [Bundle(for: type(of: self))]) else { throw ErrorCase.NoModelFound }
		let store: NSPersistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		let storetype: String = storage == nil ? NSInMemoryStoreType : storage?.pathExtension == "sqlite" ? NSSQLiteStoreType : NSBinaryStoreType
		try store.addPersistentStore(ofType: storetype, configurationName: nil, at: storage, options: nil)
		persistentStoreCoordinator = store
	}
	public required init?(coder aDecoder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}
	public override func awakeAfter(using aDecoder: NSCoder) -> Any? {
		defer {
			print("awake")
		}
		return super.awakeAfter(using: aDecoder)
	}
	public override func encode(with aCoder: NSCoder) {
		super.encode(with: aCoder)
		print("enc")
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
	func make(type: DistributionType) -> Distributor {
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
		output.input.insert(try make(commandBuffer: commandBuffer, output: output, input: input, adapters: adapters))
		commandBuffer.commit()
	}
	public func disconnect(output: Cell, input: Cell) {
		output.input.filter{$0.input.objectID == input.objectID}.forEach(remove)
	}
}
extension Context {
	func make<T: Ground>() throws -> T {
		let name: String = String(describing: T.self)
		var cache: NSManagedObject?
		func block() {
			cache = NSEntityDescription.insertNewObject(forEntityName: name, into: self)
		}
		performAndWait(block)
		guard let entity: T = cache as? T else { throw ErrorCase.InvalidEntity(name: name) }
		return entity
	}
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
	public override func save() throws {
		var encounter: Error?
		func done(_: CommandBuffer) {
			func block() {
				do {
					try super.save()
				} catch {
					encounter = error
				}
			}
			performAndWait(block)
		}
		let commandBuffer: CommandBuffer = make()
		commandBuffer.addCompletedHandler(done)
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
