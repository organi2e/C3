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
internal class Arcane: Ground {
	var μ: Variable?
	var σ: Variable?
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let scalekey: String = "scale"
	func change(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
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
		guard let μ: Variable = μ, let σ: Variable = σ else { fatalError(String(describing: self)) }
		func will() {
			willAccessValue(forKey: Arcane.locationkey)
			willAccessValue(forKey: Arcane.scalekey)
		}
		func done() {
			didAccessValue(forKey: Arcane.locationkey)
			didAccessValue(forKey: Arcane.scalekey)
		}
		context.performAndWait(will)
		handler((μ: μ.θ, σ: σ.θ))
		context.perform(done)
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
		
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
	@NSManaged var locationType: String
	@NSManaged var location: Data
	@NSManaged var scaleType: String
	@NSManaged var scale: Data
}
private extension String {
	var adapterType: AdapterType {
		guard let adapterType: AdapterType = AdapterType(rawValue: self) else { fatalError(self) }
		return adapterType
	}
}
