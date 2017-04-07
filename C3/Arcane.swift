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
	var c: Bool = false
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
			context.performAndWait(block)
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
		
		c = false
		
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func access(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		guard let μ: Variable = μ, let σ: Variable = σ else { fatalError(String(describing: self)) }
		func will(_: CommandBuffer) {
			func block() {
				willAccessValue(forKey: Arcane.locationkey)
				willAccessValue(forKey: Arcane.scalekey)
			}
			context.performAndWait(block)
		}
		func done(_: CommandBuffer) {
			func block() {
				didAccessValue(forKey: Arcane.locationkey)
				didAccessValue(forKey: Arcane.scalekey)
			}
			context.perform(block)
		}
		if !c {
			μ.refresh(commandBuffer: commandBuffer)
			σ.refresh(commandBuffer: commandBuffer)
			c = true
		}
		
		handler((μ: μ.θ, σ: σ.θ))
		
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
	
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= scale.count)
		
		μ = Variable(context: context, data: location, adapter: context.make(count: count, type: locationType.adapterType), optimizer: context.optimizerFactory(count))
		σ = Variable(context: context, data: scale, adapter: context.make(count: count, type: scaleType.adapterType), optimizer: context.optimizerFactory(count))
		
		μ?.reset(commandBuffer: commandBuffer)
		σ?.reset(commandBuffer: commandBuffer)
		
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
		locationType = AdapterType.Regular.rawValue
		scaleType = AdapterType.RegFloor.rawValue
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var locationType: String
	@NSManaged var scale: Data
	@NSManaged var scaleType: String
}
private extension String {
	var adapterType: AdapterType {
		guard let adapterType: AdapterType = AdapterType(rawValue: self) else { fatalError(self) }
		return adapterType
	}
}
