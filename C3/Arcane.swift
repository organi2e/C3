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
			encoder.label = "Arcane.Flush"
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
		                         adapter: context.make(count: count, type: locationType.adapterType),
		                         optimizer: context.optimizerFactory(count))
			
		locationCache.reset(commandBuffer: commandBuffer)
		locationCache.refresh(commandBuffer: commandBuffer)
		setPrimitiveValue(locationCache.data, forKey: "location")
		
		scaleCache = Variable(context: context, data: scale,
		                      adapter: context.make(count: count, type: scaleType.adapterType),
		                      optimizer: context.optimizerFactory(count))
			
		scaleCache.reset(commandBuffer: commandBuffer)
		scaleCache.refresh(commandBuffer: commandBuffer)
		setPrimitiveValue(scaleCache.data, forKey: "scale")
		
		commandBuffer.label = "Arcane.setup"
		commandBuffer.commit()

	}
	internal func shuffle(commandBuffer: CommandBuffer, count: Int, μ: Float = 0, σ: Float = 1) {
		func shuffle(_: CommandBuffer) {
			assert( count * MemoryLayout<Float>.size <= location.count )
			location.withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) -> Void in
				assert( MemoryLayout<Float>.size == 4 )
				assert( MemoryLayout<UInt32>.size == 4 )
				arc4random_buf(ref, location.count)
				vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(count))
				vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(count))
				cblas_sscal(Int32(count/2), 2*Float.pi, ref.advanced(by: count/2), 1)
				vvlogf(ref, ref, [Int32(count/2)])
				cblas_sscal(Int32(count/2), -2, ref, 1)
				vvsqrtf(ref, ref, [Int32(count/2)])
				vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: count/2), 2, vDSP_Length(count/4))
				vDSP_rect(ref, 2, ref, 2, vDSP_Length(count/2))
				vDSP_vsmsa(ref, 1, [σ], [μ], ref, 1, vDSP_Length(count))
			}
			assert( count * MemoryLayout<Float>.size <= scale.count )
			scale.withUnsafeMutableBytes {
				vDSP_vfill([Float(1)], $0, 1, vDSP_Length(count))
			}
		}
		commandBuffer.addCompletedHandler(shuffle)
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
		guard let adapterType: AdapterType = AdapterType(rawValue: self) else { fatalError(self) }
		return adapterType
	}
}
