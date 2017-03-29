//
//  DistributorTests.swift
//  DistributorTests
//
//  Created by Kota Nakano on 2017/03/29.
//
//
import Accelerate
import MetalKit
import simd
import XCTest
@testable import Distributor
class GaussDerivatorTests: XCTestCase {
	func testDerivateP() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let xlen: Int = 16 + Int(arc4random_uniform(240))
		let ylen: Int = 16 + Int(arc4random_uniform(240))
		let zlen: Int = 16 + Int(arc4random_uniform(240))
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: ylen), options: [])
		let Δyc: MTLBuffer = device.makeBuffer(array: uniform(count: ylen), options: [])
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let Δz: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: zlen), options: []),
			σ: device.makeBuffer(array: uniform(count: zlen), options: [])
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: xlen), options: [])
		let y: MTLBuffer = device.makeBuffer(array: uniform(count: ylen).map(sign), options: [])
		let b: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen*ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen*ylen), options: [])
		)
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen*xlen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen*xlen), options: [])
		)
		let Δw: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen*xlen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen*xlen), options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let Δc: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: zlen*ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: zlen*ylen), options: [])
		)
		let f: MTLBuffer = device.makeBuffer(array: uniform(count: ylen).map(sign), options: [])
		
		let la_one: la_object_t = la_vector_from_splat(la_splat_from_float(1, attr), la_count_t(ylen))
		let la_pμ: la_object_t = p.μ.matrix(rows: ylen, cols: 1)
		let la_pσ: la_object_t = p.σ.matrix(rows: ylen, cols: 1)
		let la_d: la_object_t = d.matrix(rows: ylen, cols: 1)
		let la_cμ: la_object_t = c.μ.matrix(rows: ylen, cols: 1)
		let la_cσ: la_object_t = c.σ.matrix(rows: ylen, cols: 1)
		let la_x: la_object_t = x.matrix(rows: xlen, cols: 1)
		let la_y: la_object_t = y.matrix(rows: ylen, cols: 1)
		let la_wμ: la_object_t = w.μ.matrix(rows: ylen, cols: xlen)
		let la_wσ: la_object_t = w.σ.matrix(rows: ylen, cols: xlen)
		let la_bμ: la_object_t = b.μ.matrix(rows: ylen, cols: ylen)
		let la_bσ: la_object_t = b.σ.matrix(rows: ylen, cols: ylen)
		
		let la_φμ: la_object_t = la_sum(la_sum(la_matrix_product(  (la_wμ),   (la_x)), la_elementwise_product(  (la_d),   (la_pμ))),   (la_cμ))
		let la_φv: la_object_t = la_sum(la_sum(la_matrix_product(sq(la_wσ), sq(la_x)), la_elementwise_product(sq(la_d), sq(la_pσ))), sq(la_cσ))
		
		let la_jμ: la_object_t = j.μ.matrix(rows: zlen, cols: ylen)
		let la_jσ: la_object_t = j.σ.matrix(rows: zlen, cols: ylen)
		
		let la_Δzμ: la_object_t = Δz.μ.matrix(rows: zlen, cols: 1)
		let la_Δzσ: la_object_t = Δz.σ.matrix(rows: zlen, cols: 1)
		let la_Δyc: la_object_t = Δyc.matrix(rows: ylen, cols: 1)
		
		let la_Δf: la_object_t = la_sum(la_sum(la_matrix_product(la_transpose(la_jμ), la_Δzμ), la_matrix_product(la_transpose(la_jσ), la_Δzσ)), la_Δyc)
		let la_gμ: la_object_t = la_matrix_from_float_buffer(
			zip(la_φμ.array, la_φv.array).map { exp(-0.5*$0.0*$0.0/$0.1) * rsqrt(Float(2.0*M_PI)*$0.1) } ,
			la_count_t(ylen), la_count_t(1), 1, hint, attr)
		let la_gσ: la_object_t = la_matrix_from_float_buffer(
			zip(la_φμ.array, la_φv.array).map { -exp(-0.5*$0.0*$0.0/$0.1) * rsqrt(Float(2.0*M_PI)) * $0.0 / $0.1 } ,
			la_count_t(ylen), la_count_t(1), 1, hint, attr)
		let la_Δφμ: la_object_t = la_elementwise_product(la_gμ, la_Δf)
		let la_Δφσ: la_object_t = la_elementwise_product(la_gσ, la_Δf)
		
		do {
			let activator: Activator = try GaussActivator.factory(device: device)(ylen, 2)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				activator.activate(commandBuffer: commandBuffer, f: f) {
					$0.collect(c: p)
				}
				activator.derivate(commandBuffer: commandBuffer, Δφ: Δφ) {
					$0.correct(j: j, Δ: Δz, count: zlen)
					$0.correct(Δ: Δyc)
				}
				activator.activate(commandBuffer: commandBuffer, f: f) {
					$0.collect(w: w, x: x, count: xlen)
					$0.collect(c: c)
					$0.collect(d: d, Φ: activator.φ(-1))
				}
				activator.derivate(commandBuffer: commandBuffer, Δφ: Δφ) {
					$0.correct(j: j, Δ: Δz, count: zlen)
					$0.correct(Δ: Δyc)
				}
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			let λbuf: Data = Data(capacity: ylen*MemoryLayout<Float>.size)
			λbuf.withUnsafeBytes {
				vvrecf(UnsafeMutablePointer<Float>(mutating: $0), activator.φ(0).σ.ref, [Int32(ylen)])
			}
			let la_λ: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer(λbuf.withUnsafeBytes{$0}, la_count_t(ylen), 1, 1, hint, attr), 0)
			//
			do {
				let derivator: Derivator = try GaussDerivator.factory(device: device)(ylen, xlen, 2)
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				derivator.derivate(commandBuffer: commandBuffer, Δ: Δw, Δφ: Δφ, φ: activator.φ(0)) {
					$0.jacobian(a: w, x: x, count: xlen)
				}
				derivator.derivate(commandBuffer: commandBuffer, Δ: Δw, Δφ: Δφ, φ: activator.φ(0)) {
					$0.jacobian(b: b, y: y, g: activator.g(0), j: derivator.j(-1))
				}
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				
				let la_jpwμ: la_object_t = la_outer_product(la_one, la_x)
				let la_jpwσ: la_object_t = la_matrix_product(la_λ, la_elementwise_product(la_wσ, la_outer_product(la_one, la_elementwise_product(la_x, la_x))))
				
				XCTAssert(la_status(la_jpwμ)==0)
				XCTAssert(la_status(la_jpwσ)==0)
				
				let la_Δjpwμ: la_object_t = la_difference(la_jpwμ, derivator.j(-1).μ.matrix(rows: ylen, cols: xlen))
				let la_Δjpwσ: la_object_t = la_difference(la_jpwσ, derivator.j(-1).σ.matrix(rows: ylen, cols: xlen))
				
				XCTAssert(la_status(la_Δjpwμ)==0)
				XCTAssert(la_status(la_Δjpwσ)==0)
				
				let rmseΔjpwμ: Float = la_norm_as_float(la_Δjpwμ, norm)
				let rmseΔjpwσ: Float = la_norm_as_float(la_Δjpwσ, norm)
				
				XCTAssert(!rmseΔjpwμ.isNaN && rmseΔjpwμ < 1e-4)
				XCTAssert(!rmseΔjpwσ.isNaN && rmseΔjpwσ < 1e-4)
				
				let la_jwμ: la_object_t = la_matrix_product(la_bμ, la_matrix_product(la_gμ.diagonale, la_jpwμ))
				let la_jwσ: la_object_t = la_matrix_product(la_λ, la_matrix_product(la_elementwise_product(la_bσ, la_bσ), la_matrix_product(la_elementwise_product(la_y, la_gσ).diagonale, la_jpwσ)))
				
				XCTAssert(la_status(la_jwμ)==0)
				XCTAssert(la_status(la_jwσ)==0)
				
				let la_Δjwμ: la_object_t = la_difference(la_jwμ, derivator.j(0).μ.matrix(rows: ylen, cols: xlen))
				let la_Δjwσ: la_object_t = la_difference(la_jwσ, derivator.j(0).σ.matrix(rows: ylen, cols: xlen))
				
				XCTAssert(la_status(la_Δjwμ)==0)
				XCTAssert(la_status(la_Δjwσ)==0)
				
				let rmseΔjwμ: Float = la_norm_as_float(la_Δjwμ, norm)
				let rmseΔjwσ: Float = la_norm_as_float(la_Δjwσ, norm)
				
				XCTAssert(!rmseΔjwμ.isNaN && rmseΔjwμ < 1e-3 )
				XCTAssert(!rmseΔjwσ.isNaN && rmseΔjwσ < 1e-3 )
			}
			//
			do {
				let derivator: GaussDerivator = try GaussDerivator.factory(device: device)(ylen, 1, 2) as! GaussDerivator
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				derivator.derivate(commandBuffer: commandBuffer, Δ: Δc, Δφ: Δφ, φ: activator.φ(0)) {
					$0.jacobian(c: c)
				}
				derivator.derivate(commandBuffer: commandBuffer, Δ: Δc, Δφ: Δφ, φ: activator.φ(0)) {
					$0.jacobian(d: d, φ: activator.φ(-1), j: derivator.j(-1))
					$0.jacobian(c: c)
				}
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				
				let la_jcμ: la_object_t = la_sum(la_elementwise_product(la_d, la_one), la_one)
				let la_jcσ: la_object_t = la_matrix_product(la_λ, la_sum(
					la_elementwise_product(la_elementwise_product(activator.φ(-1).σ.matrix(rows: ylen, cols: 1), la_elementwise_product(la_d, la_d)), la_matrix_product(la_λ, la_cσ)), la_cσ))
				
				let la_Δjcμ: la_object_t = la_difference(la_jcμ, derivator.j(0).μ.matrix(rows: ylen, cols: 1))
				let la_Δjcσ: la_object_t = la_difference(la_jcσ, derivator.j(0).σ.matrix(rows: ylen, cols: 1))
				
				XCTAssert(la_status(la_Δjcμ)==0)
				XCTAssert(la_status(la_Δjcσ)==0)
				
				let rmse_Δjcμ: Float = la_norm_as_float(la_Δjcμ, norm)
				let rmse_Δjcσ: Float = la_norm_as_float(la_Δjcσ, norm)
				
				XCTAssert( rmse_Δjcμ < 1e-4 )
				XCTAssert( rmse_Δjcσ < 1e-4 )
				
				let la_Δcμ: la_object_t = la_difference(la_elementwise_product(la_jcμ, la_Δφμ), Δc.μ.matrix(rows: ylen, cols: 1))
				let la_Δcσ: la_object_t = la_difference(la_elementwise_product(la_jcσ, la_Δφσ), Δc.σ.matrix(rows: ylen, cols: 1))
				
				XCTAssert(la_status(la_Δcμ)==0)
				XCTAssert(la_status(la_Δcσ)==0)
				
				let rmse_Δcμ: Float = la_norm_as_float(la_Δcμ, norm)
				let rmse_Δcσ: Float = la_norm_as_float(la_Δcσ, norm)
				
				XCTAssert( rmse_Δcμ < 1e-4 )
				XCTAssert( rmse_Δcσ < 1e-4 )
				
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
class GaussActivatorTests: XCTestCase {
	func testActivateP() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let xlen: Int = 16 + Int(arc4random_uniform(240))
		let ylen: Int = 16 + Int(arc4random_uniform(240))
		let zlen: Int = 16 + Int(arc4random_uniform(240))
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: ylen), options: [])
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let Δc: MTLBuffer = device.makeBuffer(array: uniform(count: ylen), options: [])
		let Δy: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen), options: [])
		)
		let Δz: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: zlen), options: []),
			σ: device.makeBuffer(array: uniform(count: zlen), options: [])
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: xlen), options: [])
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: ylen*xlen), options: []),
			σ: device.makeBuffer(array: uniform(count: ylen*xlen), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: zlen*ylen), options: []),
			σ: device.makeBuffer(array: uniform(count: zlen*ylen), options: [])
		)
		let f: MTLBuffer = device.makeBuffer(array: uniform(count: ylen), options: [])
		
		let la_pμ: la_object_t = p.μ.matrix(rows: ylen, cols: 1)
		let la_pσ: la_object_t = p.σ.matrix(rows: ylen, cols: 1)
		let la_d: la_object_t = d.matrix(rows: ylen, cols: 1)
		let la_cμ: la_object_t = c.μ.matrix(rows: ylen, cols: 1)
		let la_cσ: la_object_t = c.σ.matrix(rows: ylen, cols: 1)
		let la_x: la_object_t = x.matrix(rows: xlen, cols: 1)
		let la_wμ: la_object_t = w.μ.matrix(rows: ylen, cols: xlen)
		let la_wσ: la_object_t = w.σ.matrix(rows: ylen, cols: xlen)
		
		
		//		let la_φμ: la_object_t =   (la_cμ)
		//		let la_φv: la_object_t = sq(la_cσ)
		
		//		let la_φμ: la_object_t = la_elementwise_product(  (la_d),   (la_pμ))
		//		let la_φv: la_object_t = la_elementwise_product(sq(la_d), sq(la_pσ))
		
		//		let la_φμ: la_object_t = la_matrix_product(  (la_wμ),   (la_x))
		//		let la_φv: la_object_t = la_matrix_product(sq(la_wσ), sq(la_x))
		
		let la_φμ: la_object_t = la_sum(la_sum(la_matrix_product(  (la_wμ),   (la_x)), la_elementwise_product(  (la_d),   (la_pμ))),   (la_cμ))
		let la_φv: la_object_t = la_sum(la_sum(la_matrix_product(sq(la_wσ), sq(la_x)), la_elementwise_product(sq(la_d), sq(la_pσ))), sq(la_cσ))
		
		let la_jμ: la_object_t = j.μ.matrix(rows: zlen, cols: ylen)
		let la_jσ: la_object_t = j.σ.matrix(rows: zlen, cols: ylen)
		
		let la_Δzμ: la_object_t = Δz.μ.matrix(rows: zlen, cols: 1)
		let la_Δzσ: la_object_t = Δz.σ.matrix(rows: zlen, cols: 1)
		
		let la_Δc: la_object_t = Δc.matrix(rows: ylen, cols: 1)
		let la_Δy: la_object_t = la_sum(la_sum(la_matrix_product(la_transpose(la_jμ), la_Δzμ), la_matrix_product(la_transpose(la_jσ), la_Δzσ)), la_Δc)
		let la_gμ: la_object_t = la_matrix_from_float_buffer(
			zip(la_φμ.array, la_φv.array).map { exp(-0.5*$0.0*$0.0/$0.1) * rsqrt(Float(2.0*M_PI)*$0.1) } ,
			la_count_t(ylen), la_count_t(1), 1, hint, attr)
		let la_gσ: la_object_t = la_matrix_from_float_buffer(
			zip(la_φμ.array, la_φv.array).map { -exp(-0.5*$0.0*$0.0/$0.1) * rsqrt(Float(2.0*M_PI)) * $0.0 / $0.1 } ,
			la_count_t(ylen), la_count_t(1), 1, hint, attr)
		let la_Δyμ: la_object_t = la_elementwise_product(la_gμ, la_Δy)
		let la_Δyσ: la_object_t = la_elementwise_product(la_gσ, la_Δy)
		
		
		do {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			let activator: Activator = try GaussActivator.factory(device: device)(ylen, 3)
			activator.activate(commandBuffer: commandBuffer, f: f) {
				$0.collect(c: p)
			}
			activator.activate(commandBuffer: commandBuffer, f: f) {
				$0.collect(w: w, x: x, count: xlen)
				$0.collect(c: c)
				$0.collect(d: d, Φ: activator.φ(-1))
			}
			activator.derivate(commandBuffer: commandBuffer, Δφ: Δy) {
				$0.correct(j: j, Δ: Δz, count: zlen)
				$0.correct(Δ: Δc)
			}
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let φ: (μ: MTLBuffer, σ: MTLBuffer) = activator.φ(0)
			
			let Δφμ: la_object_t = la_difference(   φ.μ.matrix(rows: ylen, cols: 1),  la_φμ)
			let Δφv: la_object_t = la_difference(sq(φ.σ.matrix(rows: ylen, cols: 1)), la_φv)
			
			XCTAssert(la_status(Δφμ)==0)
			XCTAssert(la_status(Δφv)==0)
			
			let rmseφμ: Float = la_norm_as_float(Δφμ, norm) / Float(ylen)
			let rmseφv: Float = la_norm_as_float(Δφv, norm) / Float(ylen)
			
			XCTAssert( !rmseφμ.isNaN && rmseφμ < 1e-4 )
			XCTAssert( !rmseφv.isNaN && rmseφv < 1e-4 )
			
			let ΔΔyμ: la_object_t = la_difference(Δy.μ.matrix(rows: ylen, cols: 1), la_Δyμ)
			let ΔΔyσ: la_object_t = la_difference(Δy.σ.matrix(rows: ylen, cols: 1), la_Δyσ)
			
			XCTAssert(la_status(ΔΔyμ)==0)
			XCTAssert(la_status(ΔΔyσ)==0)
			
			let rmseΔyμ: Float = la_norm_as_float(ΔΔyμ, norm) / Float(ylen)
			let rmseΔyσ: Float = la_norm_as_float(ΔΔyσ, norm) / Float(ylen)
			
			XCTAssert( !rmseΔyμ.isNaN && rmseΔyμ < 1e-4 )
			XCTAssert( !rmseΔyσ.isNaN && rmseΔyσ < 1e-4 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
private extension MTLDevice {
	func makeBuffer<T>(array: Array<T>, options: MTLResourceOptions) -> MTLBuffer {
		return makeBuffer(bytes: array, length: array.count * MemoryLayout<T>.size, options: options)
	}
}
private extension MTLCommandQueue {
	func toArray<T>(buffer: MTLBuffer) -> Array<T> {
		let cache: MTLBuffer = device.makeBuffer(length: buffer.length, options: .storageModeShared)
		let command: MTLCommandBuffer = makeCommandBuffer()
		let encoder: MTLBlitCommandEncoder = command.makeBlitCommandEncoder()
		encoder.copy(from: buffer, sourceOffset: 0, to: cache, destinationOffset: 0, size: min(buffer.length, cache.length))
		encoder.endEncoding()
		command.commit()
		command.waitUntilCompleted()
		defer {
			cache.setPurgeableState(.empty)
		}
		return Array<T>(UnsafeBufferPointer<T>(start: UnsafePointer<T>(OpaquePointer(cache.contents())), count: cache.length/MemoryLayout<T>.size))
	}
}
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private extension MTLBuffer {
	func matrix(rows: Int, cols: Int) -> la_object_t {
		XCTAssert(rows*cols*MemoryLayout<Float>.size<=length)
		return la_matrix_from_float_buffer(UnsafePointer<Float>(OpaquePointer(contents())), la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
	}
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var buf: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: ref, count: count)
	}
	var count: Int {
		return length / MemoryLayout<Float>.size
	}
}
private extension la_object_t {
	var array: Array<Float> {
		let result: Array<Float> = Array<Float>(repeating: 0, count: Int(la_matrix_rows(self)*la_matrix_cols(self)))
		la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: result), la_matrix_cols(self), self)
		return result
	}
	var diagonale: la_object_t {
		return la_diagonal_matrix_from_vector(self, 0)
	}
}
private func uniform(count: Int, α: Float = -1, β: Float = 1) -> Array<Float> {
	let array: Array<Float> = Array<Float>(repeating: 0, count: count)
	let seeds: Array<UInt32> = Array<UInt32>(repeating: 0, count: count)
	
	arc4random_buf(UnsafeMutablePointer<UInt32>(mutating: seeds), count * MemoryLayout<UInt32>.size)
	
	vDSP_vfltu32(seeds, 1, UnsafeMutablePointer<Float>(mutating: array), 1, vDSP_Length(count))
	vDSP_vsmsa(array, 1, [(β-α)/Float(1<<16)/Float(1<<16)], [α], UnsafeMutablePointer<Float>(mutating: array), 1, vDSP_Length(count))
	
	return array
}
private func sq(_ x: la_object_t) -> la_object_t {
	return la_elementwise_product(x, x)
}
