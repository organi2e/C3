//
//  Arcane.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Adapter
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
		init(commandBuffer: CommandBuffer, data: Data, adapter: Adapter, optimizer: Optimizer) {
			let device: Device = commandBuffer.device
			φ = data.withUnsafeBytes { device.makeBuffer(bytes: $0, length: data.count, options: .storageModeShared) }
			θ = device.makeBuffer(length: data.count, options: .storageModePrivate)
			Δ = device.makeBuffer(length: data.count, options: .storageModePrivate)
			a = adapter
			o = optimizer
			o.reset(commandBuffer: commandBuffer)	
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
		var data: Data {
			return Data(bytesNoCopy: φ.contents(), count: φ.length, deallocator: .none)
		}
	}
	internal func update(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		if let μ: Variable = μ, let σ: Variable = σ {
			func will(_: CommandBuffer) {
				willChangeValue(forKey: Arcane.locationkey)
				willChangeValue(forKey: Arcane.logscalekey)
			}
			func done(_: CommandBuffer) {
				didChangeValue(forKey: Arcane.logscalekey)
				didChangeValue(forKey: Arcane.locationkey)
			}
			
			μ.flush(commandBuffer: commandBuffer)
			σ.flush(commandBuffer: commandBuffer)
			
			handler(μ: μ.Δ, σ: σ.Δ)
			
			μ.update(commandBuffer: commandBuffer)
			σ.update(commandBuffer: commandBuffer)
			
			commandBuffer.addScheduledHandler(will)
			commandBuffer.addCompletedHandler(done)
		}
	}
	internal func refresh(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		if let μ: Variable = μ, let σ: Variable = σ {
			func will(_: CommandBuffer) {
				willAccessValue(forKey: Arcane.locationkey)
				willAccessValue(forKey: Arcane.logscalekey)
			}
			func done(_: CommandBuffer) {
				didAccessValue(forKey: Arcane.logscalekey)
				didAccessValue(forKey: Arcane.locationkey)
			}
			
			μ.refresh(commandBuffer: commandBuffer)
			σ.refresh(commandBuffer: commandBuffer)
			
			handler((μ: μ.θ, σ: σ.θ))
			
			commandBuffer.addScheduledHandler(will)
			commandBuffer.addCompletedHandler(done)
		}
	}
	internal func setup(commandBuffer: CommandBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= logscale.count)
		
		μ = Structs(commandBuffer: commandBuffer, data: location, adapter: context.μadapterFactory(count), optimizer: context.optimizerFactory(count))
		σ = Structs(commandBuffer: commandBuffer, data: logscale, adapter: context.σadapterFactory(count), optimizer: context.optimizerFactory(count))
		
		setPrimitiveValue(μ?.data, forKey: Arcane.locationkey)
		setPrimitiveValue(σ?.data, forKey: Arcane.logscalekey)
		
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
