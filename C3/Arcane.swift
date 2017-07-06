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
import simd
internal class Arcane: Ground {
	
}
extension Arcane {
	private class Variable: NSObject {
		internal let Δ: Buffer
		internal let θ: Buffer
		internal let φ: Buffer
		private let a: Adapter
		private let o: Optimizer
		init(context: Context, data: Data, adapter: Adapter, optimizer: Optimizer) {
			Δ = context.make(length: data.count, options: .storageModePrivate)
			θ = context.make(length: data.count, options: .storageModePrivate)
			φ = context.make(data: data, options: .storageModeShared)
			a = adapter
			o = optimizer
			super.init()
		}
		func flush(commandBuffer: CommandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: Δ, range: NSRange(location: 0, length: Δ.length), value: 0)
			encoder.label = #function
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
	func change(commandBuffer: CommandBuffer, handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		
		func will(_: MTLCommandBuffer) {
			willChangeValue(forKey: "location")
			willChangeValue(forKey: "scale")
		}
		func done(_: MTLCommandBuffer) {
			didChangeValue(forKey: "location")
			didChangeValue(forKey: "scale")
		}
		
		locationCache.flush(commandBuffer: commandBuffer)
		scaleCache.flush(commandBuffer: commandBuffer)
		
		handler(μ: locationCache.Δ, σ: scaleCache.Δ)
		
		locationCache.update(commandBuffer: commandBuffer)
		scaleCache.update(commandBuffer: commandBuffer)
		
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func fixing(commandBuffer: CommandBuffer) {
		
		func will(_: MTLCommandBuffer) {
			willAccessValue(forKey: "location")
			willAccessValue(forKey: "scale")
		}
		func done(_: MTLCommandBuffer) {
			didAccessValue(forKey: "location")
			didAccessValue(forKey: "scale")
		}
		
		locationCache.refresh(commandBuffer: commandBuffer)
		scaleCache.refresh(commandBuffer: commandBuffer)
	
		commandBuffer.addScheduledHandler(will)
		commandBuffer.addCompletedHandler(done)
		
	}
	func access(handler: ((μ: Buffer, σ: Buffer)) -> Void) {
		handler((μ: locationCache.θ, σ: scaleCache.θ))
	}
	func setup(context: Context, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= location.count)
		assert( count * MemoryLayout<Float>.size <= scale.count)
		
		let commandBuffer: CommandBuffer = context.make()
		
		locationCache = Variable(context: context, data: location,
		                         adapter: context.make(type: locationType.adapterType, count: count),
		                         optimizer: context.optimizerFactory(count))
		locationCache.reset(commandBuffer: commandBuffer)
		locationCache.refresh(commandBuffer: commandBuffer)
		setPrimitiveValue(locationCache.data, forKey: "location")
		
		scaleCache = Variable(context: context, data: scale,
		                      adapter: context.make(type: scaleType.adapterType, count: count),
		                      optimizer: context.optimizerFactory(count))
		scaleCache.reset(commandBuffer: commandBuffer)
		scaleCache.refresh(commandBuffer: commandBuffer)
		setPrimitiveValue(scaleCache.data, forKey: "scale")
		
		commandBuffer.label = #function
		commandBuffer.commit()

	}
	@NSManaged private var locationCache: Variable
	@NSManaged private var scaleCache: Variable
}
extension Arcane {
	@NSManaged internal var locationType: String
	@NSManaged internal var location: Data
	@NSManaged internal var scaleType: String
	@NSManaged internal var scale: Data
}
private extension String {
	var adapterType: AdapterType {
		guard let adapterType: AdapterType = AdapterType(rawValue: self) else { fatalError(#function) }
		return adapterType
	}
}
