//
//  Arcane.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Adapter
import Distributor
import Optimizer
internal class Arcane: ManagedObject {
	var μ: Variable?
	var σ: Variable?
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let logscalekey: String = "logscale"
	private struct Structs: Variable {
		internal let θ: Buffer
		internal let Δ: Buffer
		private let φ: Buffer
		private let a: Adapter
		private let o: Optimizer
		init(context: Context, data: Data, adapter: Adapter, optimizer: Optimizer) {
			φ = context.make(data: data, options: .storageModeShared)
			θ = context.make(length: data.count, options: .storageModePrivate)
			Δ = context.make(length: data.count, options: .storageModePrivate)
			a = adapter
			o = optimizer
		}
		func flush(commandBuffer: CommandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δ, range: NSRange(location: 0, length: Δ.length), value: 0)
			encoder.endEncoding()
		}
		func refresh(commandBuffer: CommandBuffer) {
			a.generate(commandBuffer: commandBuffer, θ: θ, φ: φ)
		}
		func update(commandBuffer: CommandBuffer) {
			a.gradient(commandBuffer: commandBuffer, Δ: Δ, θ: θ, φ: φ)
			o.optimize(commandBuffer: commandBuffer, θ: φ, Δ: Δ)
		}
		func reset(commandBuffer: CommandBuffer) {
			o.reset(commandBuffer: commandBuffer)
		}
		var data: Data {
			return Data(bytesNoCopy: φ.contents(), count: φ.length, deallocator: .none)
		}
	}
	func update(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		if let μ: Variable = μ, let σ: Variable = σ {
			func will(_: CommandBuffer) {
				func block() {
					willChangeValue(forKey: Arcane.locationkey)
					willChangeValue(forKey: Arcane.logscalekey)
				}
				context.perform(block)
			}
			func done(_: CommandBuffer) {
				func block() {
					didChangeValue(forKey: Arcane.logscalekey)
					didChangeValue(forKey: Arcane.locationkey)
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
		} else {
			assertionFailure()
		}
	}
	func access(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		if let μ: Variable = μ, let σ: Variable = σ {
			func will(_: CommandBuffer) {
				func block() {
					willAccessValue(forKey: Arcane.locationkey)
					willAccessValue(forKey: Arcane.logscalekey)
				}
				context.perform(block)
			}
			func done(_: CommandBuffer) {
				func block() {
					didAccessValue(forKey: Arcane.logscalekey)
					didAccessValue(forKey: Arcane.locationkey)
				}
				context.perform(block)
			}
			
			μ.refresh(commandBuffer: commandBuffer)
			σ.refresh(commandBuffer: commandBuffer)
			
			handler((μ: μ.θ, σ: σ.θ))
			
			commandBuffer.addScheduledHandler(will)
			commandBuffer.addCompletedHandler(done)
		} else {
			assertionFailure()
		}
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
	
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= logscale.count)
		
		μ = Structs(context: context, data: location, adapter: context.μadapterFactory(count), optimizer: context.optimizerFactory(count))
		σ = Structs(context: context, data: logscale, adapter: context.σadapterFactory(count), optimizer: context.optimizerFactory(count))
		
		μ?.reset(commandBuffer: commandBuffer)
		σ?.reset(commandBuffer: commandBuffer)
		
		func block() {
			setPrimitiveValue(μ?.data, forKey: Arcane.locationkey)
			setPrimitiveValue(σ?.data, forKey: Arcane.logscalekey)
		}
		context.perform(block)
		
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
