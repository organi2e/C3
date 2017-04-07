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
	private static let lifetimekey: String = "lifetime"
	private static let locationkey: String = "location"
	private static let scalekey: String = "scale"
	func change(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		willAccessValue(forKey: Arcane.lifetimekey)
		defer { didAccessValue(forKey: Arcane.lifetimekey) }
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
		
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func fixing(commandBuffer: CommandBuffer) {
		willAccessValue(forKey: Arcane.lifetimekey)
		defer { didAccessValue(forKey: Arcane.lifetimekey) }
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
		
		μ.refresh(commandBuffer: commandBuffer)
		σ.refresh(commandBuffer: commandBuffer)
		
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)

	}
	func access(handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		willAccessValue(forKey: Arcane.lifetimekey)
		defer { didAccessValue(forKey: Arcane.lifetimekey) }
		guard let μ: Variable = μ, let σ: Variable = σ else { fatalError(String(describing: self)) }
		handler((μ: μ.θ, σ: σ.θ))
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
		
		willAccessValue(forKey: Arcane.lifetimekey)
		defer { didAccessValue(forKey: Arcane.lifetimekey) }
		
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= scale.count)
		
		μ = Variable(context: context, data: location, adapter: context.make(count: count, type: locationType.adapterType), optimizer: context.optimizerFactory(count))
		σ = Variable(context: context, data: scale, adapter: context.make(count: count, type: scaleType.adapterType), optimizer: context.optimizerFactory(count))
		
		setPrimitiveValue(μ?.data, forKey: Arcane.locationkey)
		setPrimitiveValue(σ?.data, forKey: Arcane.scalekey)
		
		μ?.reset(commandBuffer: commandBuffer)
		σ?.reset(commandBuffer: commandBuffer)
		
		μ?.refresh(commandBuffer: commandBuffer)
		σ?.refresh(commandBuffer: commandBuffer)
		
	}
}
extension Arcane {
	override func awakeFromInsert() {
		super.awakeFromInsert()
		lifetime = true
		locationType = AdapterType.Regular.rawValue
		location = Data()
		scaleType = AdapterType.RegFloor.rawValue
		scale = Data()
	}
}
extension Arcane {
	@NSManaged var lifetime: Bool
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
