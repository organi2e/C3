//
//  Arcane.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import Adapter
import Distributor
import Optimizer
internal class Arcane: ManagedObject {
	var μ: Variable?
	var σ: Variable?
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let scalekey: String = "scale"
	func update(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		guard let μ: Variable = μ, let σ: Variable = σ else { fatalError(String(describing: self)) }
		func will(_: CommandBuffer) {
			func block() {
				willChangeValue(forKey: Arcane.locationkey)
				willChangeValue(forKey: Arcane.scalekey)
			}
			context.perform(block)
		}
		func done(_: CommandBuffer) {
			func block() {
				didChangeValue(forKey: Arcane.locationkey)
				didChangeValue(forKey: Arcane.scalekey)
			}
			context.perform(block)
		}
			
		μ.flush(commandBuffer: commandBuffer)
		σ.flush(commandBuffer: commandBuffer)
			
		handler(μ: μ.Δ, σ: σ.Δ)
			
		μ.update(commandBuffer: commandBuffer)
		σ.update(commandBuffer: commandBuffer)
			
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func refresh(commandBuffer: CommandBuffer) {
		μ?.refresh(commandBuffer: commandBuffer)
		σ?.refresh(commandBuffer: commandBuffer)
	}
	func access(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		guard let μ: Variable = μ, let σ: Variable = σ else { fatalError(String(describing: self)) }
		/*
		func will(_: CommandBuffer) {
			func block() {
				willAccessValue(forKey: Arcane.locationkey)
				willAccessValue(forKey: Arcane.logscalekey)
			}
			context.performAndWait(block)
		}
		func done(_: CommandBuffer) {
			func block() {
				didAccessValue(forKey: Arcane.logscalekey)
				didAccessValue(forKey: Arcane.locationkey)
			}
			context.perform(block)
		}
		*/
		
		handler((μ: μ.θ, σ: σ.θ))
			
		//commandBuffer.addScheduledHandler(will)
		//commandBuffer.addCompletedHandler(done)
		
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
	
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= scale.count)
		
		guard let locationAdapter: AdapterType = AdapterType(rawValue: locationType) else { fatalError(locationType) }
		guard let scaleAdapter: AdapterType = AdapterType(rawValue: scaleType) else { fatalError(scaleType) }
		
		μ = Variable(context: context, data: location, adapter: context.make(count: count, type: locationAdapter), optimizer: context.optimizerFactory(count))
		σ = Variable(context: context, data: scale, adapter: context.make(count: count, type: scaleAdapter), optimizer: context.optimizerFactory(count))
		
		μ?.reset(commandBuffer: commandBuffer)
		σ?.reset(commandBuffer: commandBuffer)
		
		μ?.load(commandBuffer: commandBuffer)
		σ?.load(commandBuffer: commandBuffer)
		
		func block() {
			setPrimitiveValue(μ?.data, forKey: Arcane.locationkey)
			setPrimitiveValue(σ?.data, forKey: Arcane.scalekey)
		}
		context.perform(block)
		
	}
}
extension Arcane {
	override func awakeFromInsert() {
		super.awakeFromInsert()
		locationType = AdapterType.Linear.rawValue
		scaleType = AdapterType.Linear.rawValue
	}
	override func willSave() {
		super.willSave()
		let commandBuffer: CommandBuffer = context.make()
		μ?.save(commandBuffer: commandBuffer)
		σ?.save(commandBuffer: commandBuffer)
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
	}
	override func didSave() {
		let commandBuffer: CommandBuffer = context.make()
		μ?.load(commandBuffer: commandBuffer)
		σ?.load(commandBuffer: commandBuffer)
		commandBuffer.commit()
		super.didSave()
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var locationType: String
	@NSManaged var scale: Data
	@NSManaged var scaleType: String
}
