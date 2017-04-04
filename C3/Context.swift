//
//  Context.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import CoreData
import Metal

import Adapter
import Distributor
import Optimizer

public class Context: NSManagedObjectContext {
	let device: Device
	let queue: CommandQueue
	let gaussDistributor: Distributor
	let μadapterFactory: (Int) -> Adapter
	let σadapterFactory: (Int) -> Adapter
	let optimizerFactory: (Int) -> Optimizer
	enum ErrorCase: Error, CustomStringConvertible {
		case InvalidContext
		case InvalidEntity(name: String)
		case NoModelFound
		case NoDeviceFound
		var description: String {
			switch self {
			case let .InvalidEntity(name):
				return "Invalid entity\(name)"
			case .InvalidContext:
				return "This context is invalid"
			case .NoModelFound:
				return "No CoreData definition was found"
			case .NoDeviceFound:
				return "No available Metal device found"
			}
		}
	}
	public init(storage: URL? = nil,
	            adapter: (μ: (MTLDevice)throws->(Int)->Adapter, σ: (MTLDevice)throws->(Int)->Adapter) = (μ: Linear.adapter(), σ: Linear.adapter()),
	            optimizer: (MTLDevice)throws->(Int) -> Optimizer = SGD.factory(),
	            concurrencyType: NSManagedObjectContextConcurrencyType = .privateQueueConcurrencyType) throws {
		guard let mtl: Device = MTLCreateSystemDefaultDevice() else { throw ErrorCase.NoDeviceFound }
		device = mtl
		queue = device.makeCommandQueue()
		μadapterFactory = try adapter.μ(device)
		σadapterFactory = try adapter.σ(device)
		optimizerFactory = try optimizer(device)
		gaussDistributor = try GaussDistributor(device: device)
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
}
extension Context {
	public override func save() throws {
		var encounter: Error?
		let commandBuffer: CommandBuffer = make()
		func complete(_: CommandBuffer) {
			do {
				try super.save()
			} catch {
				encounter = error
			}
		}
		commandBuffer.addCompletedHandler(complete)
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
		if let error: Error = encounter {
			throw error
		}
	}
	/*
	public func save(handler: @escaping (Error) -> Void) {
		let commandBuffer: CommandBuffer = make()
		func complete(_: CommandBuffer) {
			do {
				try save()
			} catch let error {
				handler(error)
			}
		}
		commandBuffer.addCompletedHandler(complete)
		commandBuffer.commit()
	}*/
}
extension Context {
	func make(length: Int, options: MTLResourceOptions = []) -> Buffer {
		return device.makeBuffer(length: length, options: options)
	}
	func make<T>(array: Array<T>, options: MTLResourceOptions = []) -> Buffer {
		return device.makeBuffer(bytes: array, length: array.count * MemoryLayout<T>.size, options: options)
	}
	func make() -> CommandBuffer {
		return queue.makeCommandBuffer()
	}
	func make<T: ManagedObject>() throws -> T {
		let name: String = String(describing: T.self)
		guard let entity: T = NSEntityDescription.insertNewObject(forEntityName: name, into: self) as? T else { throw ErrorCase.InvalidEntity(name: name) }
		return entity
	}
}
public typealias ManagedObject = NSManagedObject
internal extension ManagedObject {
	var context: Context {
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCase.InvalidContext.description) }
		return context
	}
}
internal protocol Variable {
	func flush(commandBuffer: CommandBuffer)
	func update(commandBuffer: CommandBuffer)
	func refresh(commandBuffer: CommandBuffer)
	var θ: Buffer { get }
	var Δ: Buffer { get }
	var data: Data { get }
}
internal typealias Device = MTLDevice
internal typealias Buffer = MTLBuffer
internal typealias CommandQueue = MTLCommandQueue
internal typealias CommandBuffer = MTLCommandBuffer
internal typealias ManagedObjectContext = NSManagedObjectContext
internal typealias BlitCommandEncoder = MTLBlitCommandEncoder
internal typealias ComputeCommandEncoder = MTLComputeCommandEncoder
