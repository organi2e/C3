//
//  GaussDistributorTests.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/30.
//
//
import Accelerate
import Metal
import simd
import XCTest
@testable import Distributor
class GaussDistributorTests: XCTestCase {
	func testAVφ() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width*refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width*refer), options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		
		let la_wμ: la_object_t = w.μ.matrix(rows: width, cols: refer)
		let la_wσ: la_object_t = w.σ.matrix(rows: width, cols: refer)
		let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
		let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
		let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: 1)
		let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: 1)
		let la_d: la_object_t = d.matrix(rows: width, cols: 1)
		let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, v: χ, g: g, φ: φ, count: width) {
				$0.collect(w: w, x: x, count: refer)
				$0.collect(c: c)
				$0.collect(d: d, φ: p)
			}
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let la_φμ: la_object_t = [
				la_matrix_product(la_wμ, la_x),
				la_elementwise_product(la_d, la_pμ),
				la_cμ
				].reduce(la_splat_from_float(0, attr), la_sum)
			let la_φv: la_object_t = [
				la_matrix_product(la_elementwise_product(la_wσ, la_wσ), la_elementwise_product(la_x, la_x)),
				la_elementwise_product(la_elementwise_product(la_d, la_d), la_elementwise_product(la_pσ, la_pσ)),
				la_elementwise_product(la_cσ, la_cσ)
				].reduce(la_splat_from_float(0, attr), la_sum)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φv) == 0 )
			
			let la_Δφμ: la_object_t = la_difference(la_φμ, φ.μ.matrix(rows: width, cols: 1))
			let la_Δφv: la_object_t = la_difference(la_φv, la_elementwise_product(φ.σ.matrix(rows: width, cols: 1), φ.σ.matrix(rows: width, cols: 1)))
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφv) == 0 )
			
			XCTAssert( 0 < la_norm_as_float(la_φμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_φv, norm) )
			
			let rmseφμ: Float = la_norm_as_float(la_Δφμ, norm) * rsqrt(Float(width))
			let rmseφv: Float = la_norm_as_float(la_Δφv, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseφμ < 1e-4 )
			XCTAssert( rmseφv < 1e-4 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testAV() {
		let width: Int = 8192 + Int(arc4random_uniform(8192))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let dμ: Float = Float(arc4random_uniform(240)) + 16
		let dσ: Float = Float(arc4random_uniform(240)) + 16
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: Array<Float>(repeating: dμ, count: width), options: []),
			σ: device.makeBuffer(array: Array<Float>(repeating: dσ, count: width), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, v: χ, g: g, φ: φ, count: width) {
				$0.collect(c: c)
			}
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			var eμ: Float = 0.0
			var eσ: Float = 0.0
			
			vDSP_normalize(χ.ref, 1, nil, 0, &eμ, &eσ, vDSP_Length(width))
			
			XCTAssert( abs(eμ-dμ) < 1 )
			XCTAssert( abs(log(eσ/dσ)) < 1e-2 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testAFφ() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width*refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width*refer), options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		
		let la_wμ: la_object_t = w.μ.matrix(rows: width, cols: refer)
		let la_wσ: la_object_t = w.σ.matrix(rows: width, cols: refer)
		let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
		let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
		let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: 1)
		let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: 1)
		let la_d: la_object_t = d.matrix(rows: width, cols: 1)
		let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, f: χ, g: g, φ: φ, count: width) {
				$0.collect(w: w, x: x, count: refer)
				$0.collect(c: c)
				$0.collect(d: d, φ: p)
			}
			commandBuffer.commit()
			
			let la_φμ: la_object_t = [
				la_matrix_product(la_wμ, la_x),
				la_elementwise_product(la_d, la_pμ),
				la_cμ
				].reduce(la_splat_from_float(0, attr), la_sum)
			let la_φv: la_object_t = [
				la_matrix_product(la_elementwise_product(la_wσ, la_wσ), la_elementwise_product(la_x, la_x)),
				la_elementwise_product(la_elementwise_product(la_d, la_d), la_elementwise_product(la_pσ, la_pσ)),
				la_elementwise_product(la_cσ, la_cσ)
				].reduce(la_splat_from_float(0, attr), la_sum)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φv) == 0 )
			
			let la_Δφμ: la_object_t = la_difference(la_φμ, φ.μ.matrix(rows: width, cols: 1))
			let la_Δφv: la_object_t = la_difference(la_φv, la_elementwise_product(φ.σ.matrix(rows: width, cols: 1),
			                                                                         φ.σ.matrix(rows: width, cols: 1)))
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφv) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_φμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_φv, norm) )
			
			let rmseφμ: Float = la_norm_as_float(la_Δφμ, norm) * rsqrt(Float(width))
			let rmseφv: Float = la_norm_as_float(la_Δφv, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseφμ < 1e-5 )
			XCTAssert( rmseφv < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testAF() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width*refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width*refer), options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		
		let la_wμ: la_object_t = w.μ.matrix(rows: width, cols: refer)
		let la_wσ: la_object_t = w.σ.matrix(rows: width, cols: refer)
		let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
		let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
		let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: 1)
		let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: 1)
		let la_d: la_object_t = d.matrix(rows: width, cols: 1)
		let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, f: χ, g: g, φ: φ, count: width) {
				$0.collect(w: w, x: x, count: refer)
				$0.collect(c: c)
				$0.collect(d: d, φ: p)
			}
			commandBuffer.commit()
			
			let la_φμ: la_object_t = [
				la_matrix_product(la_wμ, la_x),
				la_elementwise_product(la_d, la_pμ)
				].reduce(la_cμ, la_sum)
			let la_φv: la_object_t = [
				la_matrix_product(la_elementwise_product(la_wσ, la_wσ), la_elementwise_product(la_x, la_x)),
				la_elementwise_product(la_elementwise_product(la_d, la_d), la_elementwise_product(la_pσ, la_pσ))
				].reduce(la_elementwise_product(la_cσ, la_cσ), la_sum)
			let la_χ: la_object_t = la_matrix_from_float_buffer(zip(la_φμ.array, la_φv.array).map{ fma(erf($0.0*rsqrt(2.0*$0.1)), 0.5, 0.5) }, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φv) == 0 )
			XCTAssert( la_status(la_χ) == 0 )
			
			let la_Δφμ: la_object_t = la_difference(la_φμ, φ.μ.matrix(rows: width, cols: 1))
			let la_Δφv: la_object_t = la_difference(la_φv, la_elementwise_product(φ.σ.matrix(rows: width, cols: 1), φ.σ.matrix(rows: width, cols: 1)))
			let la_Δχ: la_object_t = la_difference(la_χ, χ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφv) == 0 )
			XCTAssert( la_status(la_Δχ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_φμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_φv, norm) )
			XCTAssert( 0 < la_norm_as_float(la_χ, norm) )
			
			let rmseφμ: Float = la_norm_as_float(la_Δφμ, norm) * rsqrt(Float(width))
			let rmseφv: Float = la_norm_as_float(la_Δφv, norm) * rsqrt(Float(width))
			let rmseχ: Float = la_norm_as_float(la_Δχ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseφμ < 1e-5 )
			XCTAssert( rmseφv < 1e-5 )
			XCTAssert( rmseχ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testDF() {
		let width: Int = 16 + Int(arc4random_uniform(240))
//		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: 0, β: 1), options: []),
			σ: device.makeBuffer(array: uniform(count: width, α: 0, β: 1), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: -1, β: 1), options: []),
			σ: device.makeBuffer(array: uniform(count: width, α: -1, β: 1), options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * MemoryLayout<Float>.stride, options: [])
		)
		let χ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let f: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let ϝ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		do {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			let distributor: Distributor = try GaussDistributor(device: device)
			distributor.activate(commandBuffer: commandBuffer, f: f, g: g, φ: φ, count: width) {
				$0.collect(c: c)
			}
			distributor.derivate(commandBuffer: commandBuffer, Δφ: Δφ, f: f, g: g, φ: φ, count: width) {
				$0.correct(φ: φ, f: ϝ)
			}
			commandBuffer.commit()
			
			let buf_p: Array<Float> = zip(c.μ.buf, c.σ.buf).map(Φ)
			let la_p: la_object_t = la_matrix_from_float_buffer(buf_p, la_count_t(width), 1, 1, hint, attr)
			
			let buf_r: Array<Float> = buf_p.map { 1.0 / $0 / ( 1.0 - $0 ) }
			let la_r: la_object_t = la_matrix_from_float_buffer(buf_r, la_count_t(width), 1, 1, hint, attr)
			
			let la_gμ: la_object_t = la_matrix_from_float_buffer(zip(c.μ.buf, c.σ.buf).map { exp( -0.5 * ( $0.0 * $0.0 ) / ( $0.1 * $0.1 ) ) * rsqrt(2.0*Float.pi) / $0.1 }, la_count_t(width), 1, 1, hint, attr)
			let la_gσ: la_object_t = la_matrix_from_float_buffer(zip(c.μ.buf, c.σ.buf).map { exp( -0.5 * ( $0.0 * $0.0 ) / ( $0.1 * $0.1 ) ) * rsqrt(2.0*Float.pi) / $0.1 / $0.1 * -$0.0 }, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_gμ) == 0 )
			XCTAssert( la_status(la_gσ) == 0 )
			
			let la_Δgμ: la_object_t = la_difference(la_gμ, g.μ.matrix(rows: width, cols: 1))
			let la_Δgσ: la_object_t = la_difference(la_gσ, g.σ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δgμ) == 0 )
			XCTAssert( la_status(la_Δgσ) == 0 )
			
			let la_χ: la_object_t = χ.matrix(rows: width, cols: 1)
			let la_ϝ: la_object_t = ϝ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_χ) == 0 )
			XCTAssert( la_status(la_ϝ) == 0 )
			
			let la_Δ: la_object_t = la_matrix_from_float_buffer([
				la_difference(la_p, la_ϝ)
			].reduce(la_splat_from_float(0, attr), la_sum).array, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_Δ) == 0 )
			
			let la_Δφμ: la_object_t = la_matrix_product(la_elementwise_product(la_gμ, la_r).diagonale, la_Δ)
			let la_Δφσ: la_object_t = la_matrix_product(la_elementwise_product(la_gσ, la_r).diagonale, la_Δ)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_ΔΔφμ: la_object_t = la_difference(la_Δφμ, Δφ.μ.matrix(rows: width, cols: 1))
			let la_ΔΔφσ: la_object_t = la_difference(la_Δφσ, Δφ.σ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_ΔΔφμ) == 0 )
			XCTAssert( la_status(la_ΔΔφσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_gμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_gσ, norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_Δφμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Δφσ, norm) )
			
			let rmsegμ: Float = la_norm_as_float(la_Δgμ, norm) * rsqrt(Float(width))
			let rmsegσ: Float = la_norm_as_float(la_Δgσ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmsegμ < 1e-6 )
			XCTAssert( rmsegσ < 1e-6 )
			
			let rmseΔφμ: Float = la_norm_as_float(la_ΔΔφμ, norm) * rsqrt(Float(width))
			let rmseΔφσ: Float = la_norm_as_float(la_ΔΔφσ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseΔφμ < 1e-2 )
			XCTAssert( rmseΔφσ < 1e-2 )
			
			print(rmseΔφμ,rmseΔφσ)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGX() {
		let width: Int = 32 + Int(arc4random_uniform(224))
		let refer: Int = 32 + Int(arc4random_uniform(224))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δx: MTLBuffer = device.makeBuffer(length: refer * MemoryLayout<Float>.stride, options: [])
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.gradient(commandBuffer: commandBuffer, Δx: Δx, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.connect(x: x, a: a)
				$0.connect(φ: φ, d: d, j: p)
			}
			commandBuffer.commit()
			
			let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
			
			XCTAssert( la_status(la_x) == 0 )
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: refer)
			let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_pμ) == 0 )
			XCTAssert( la_status(la_pσ) == 0 )
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_aμ: la_object_t = a.μ.matrix(rows: width, cols: refer)
			let la_aσ: la_object_t = a.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_aμ) == 0 )
			XCTAssert( la_status(la_aσ) == 0 )
			
			let la_jμ: la_object_t = la_sum(
				la_scale_with_float(la_aμ, 1),
				la_scale_with_float(la_matrix_product(la_d.diagonale, la_pμ), 1)
			)
			let la_jσ: la_object_t = la_matrix_product(
				la_matrix_from_float_buffer(φ.σ.buf.map(recip), la_count_t(width), 1, 1, hint, attr).diagonale,
				la_sum(la_scale_with_float(la_matrix_product(la_elementwise_product(la_aσ, la_aσ), la_x.diagonale), 1),
				       la_scale_with_float(la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_pσ), 1)
				)
			)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 < normjσ )
			
			commandBuffer.waitUntilCompleted()
			
			let rmsejμ: Float = la_norm_as_float(la_Δjμ, norm) * rsqrt(Float(width*refer))
			let rmsejσ: Float = la_norm_as_float(la_Δjσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmsejμ < 1e-5 )
			XCTAssert( rmsejσ < 1e-5 )
			
			let la_Δx: la_object_t = la_sum(
				la_matrix_product(la_transpose(la_jμ), la_Δφμ),
				la_matrix_product(la_transpose(la_jσ), la_Δφσ)
			)
			
			XCTAssert( la_status(la_Δx) == 0 )
			
			let normΔx: Float = la_norm_as_float(la_Δx, norm)
			
			XCTAssert( 0 < normΔx )
			
			let la_ΔΔx: la_object_t = la_difference(la_Δx, Δx.matrix(rows: refer, cols: 1))
			
			XCTAssert( la_status(la_ΔΔx) == 0 )
			
			let rmseΔx: Float = la_norm_as_float(la_ΔΔx, norm) * rsqrt(Float(refer))
			
			XCTAssert( rmseΔx < 1e-4 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGA() {
		let width: Int = 32 + Int(arc4random_uniform(224))
		let refer: Int = 32 + Int(arc4random_uniform(224))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δθ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.gradient(commandBuffer: commandBuffer, Δθ: Δθ, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.connect(a: a, x: x)
				$0.connect(φ: φ, d: d, j: p)
			}
			commandBuffer.commit()
			
			let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
			
			XCTAssert( la_status(la_x) == 0 )
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: refer)
			let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_pμ) == 0 )
			XCTAssert( la_status(la_pσ) == 0 )
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_aμ: la_object_t = a.μ.matrix(rows: width, cols: refer)
			let la_aσ: la_object_t = a.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_aμ) == 0 )
			XCTAssert( la_status(la_aσ) == 0 )
			
			let la_jμ: la_object_t = la_sum(
				la_scale_with_float(la_outer_product(la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(width), 1), la_x), 1),
				la_scale_with_float(la_matrix_product(la_d.diagonale, la_pμ), 1)
			)
			let la_jσ: la_object_t = la_matrix_product(
				la_matrix_from_float_buffer(φ.σ.buf.map(recip), la_count_t(width), 1, 1, hint, attr).diagonale,
				la_sum(la_scale_with_float(la_matrix_product(la_aσ, la_elementwise_product(la_x, la_x).diagonale), 1),
				       la_scale_with_float(la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_pσ), 1)
				)
			)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 < normjσ )
			
			commandBuffer.waitUntilCompleted()
			
			let rmsejμ: Float = la_norm_as_float(la_Δjμ, norm) * rsqrt(Float(width*refer))
			let rmsejσ: Float = la_norm_as_float(la_Δjσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmsejμ < 1e-5 )
			XCTAssert( rmsejσ < 1e-5 )
			
			let la_Δθμ: la_object_t = la_matrix_product(la_Δφμ.diagonale, la_jμ)
			let la_Δθσ: la_object_t = la_matrix_product(la_Δφσ.diagonale, la_jσ)
			
			XCTAssert( la_status(la_Δθμ) == 0 )
			XCTAssert( la_status(la_Δθσ) == 0 )
			
			let normΔθμ: Float = la_norm_as_float(la_Δθμ, norm)
			let normΔθσ: Float = la_norm_as_float(la_Δθσ, norm)
			
			XCTAssert( 0 < normΔθμ )
			XCTAssert( 0 < normΔθσ )
			
			let la_ΔΔθμ: la_object_t = la_difference(la_Δθμ, Δθ.μ.matrix(rows: width, cols: refer))
			let la_ΔΔθσ: la_object_t = la_difference(la_Δθσ, Δθ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΔθμ) == 0 )
			XCTAssert( la_status(la_ΔΔθσ) == 0 )
			
			let rmseΔθμ: Float = la_norm_as_float(la_ΔΔθμ, norm) * rsqrt(Float(width*refer))
			let rmseΔθσ: Float = la_norm_as_float(la_ΔΔθσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΔθμ < 1e-5 )
			XCTAssert( rmseΔθσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGC() {
		let width: Int = 32 + Int(arc4random_uniform(224))
		let refer: Int = 1 + Int(arc4random_uniform(255))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δθ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width).map(sign), options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.gradient(commandBuffer: commandBuffer, Δθ: Δθ, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.connect(c: c)
				$0.connect(φ: φ, d: d, j: p)
			}
			commandBuffer.commit()
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: refer)
			let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_pμ) == 0 )
			XCTAssert( la_status(la_pσ) == 0 )
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
			let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_cμ) == 0 )
			XCTAssert( la_status(la_cσ) == 0 )
			
			let la_jμ: la_object_t = la_sum(
				la_scale_with_float(la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(width), la_count_t(refer)), 1),
				la_scale_with_float(la_matrix_product(la_d.diagonale, la_pμ), 1)
			)
			let la_jσ: la_object_t = la_matrix_product(
				la_matrix_from_float_buffer(φ.σ.buf.map(recip), la_count_t(width), 1, 1, hint, attr).diagonale,
				la_sum(la_scale_with_float(la_outer_product(la_cσ, la_vector_from_splat(la_splat_from_float(1, attr), la_count_t(refer))), 1),
				       la_scale_with_float(la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_pσ), 1)
				)
			)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 < normjσ )
			
			commandBuffer.waitUntilCompleted()
			
			let rmsejμ: Float = la_norm_as_float(la_Δjμ, norm) * rsqrt(Float(width*refer))
			let rmsejσ: Float = la_norm_as_float(la_Δjσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmsejμ < 1e-6 )
			XCTAssert( rmsejσ < 1e-6 )
			
			let la_Δθμ: la_object_t = la_matrix_product(la_Δφμ.diagonale, la_jμ)
			let la_Δθσ: la_object_t = la_matrix_product(la_Δφσ.diagonale, la_jσ)
			
			XCTAssert( la_status(la_Δθμ) == 0 )
			XCTAssert( la_status(la_Δθσ) == 0 )
			
			let normΔθμ: Float = la_norm_as_float(la_Δθμ, norm)
			let normΔθσ: Float = la_norm_as_float(la_Δθσ, norm)
			
			XCTAssert( 0 < normΔθμ )
			XCTAssert( 0 < normΔθσ )
			
			let la_ΔΔθμ: la_object_t = la_difference(la_Δθμ, Δθ.μ.matrix(rows: width, cols: refer))
			let la_ΔΔθσ: la_object_t = la_difference(la_Δθσ, Δθ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΔθμ) == 0 )
			XCTAssert( la_status(la_ΔΔθσ) == 0 )
			
			let rmseΔθμ: Float = la_norm_as_float(la_ΔΔθμ, norm) * rsqrt(Float(width*refer))
			let rmseΔθσ: Float = la_norm_as_float(la_ΔΔθσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΔθμ < 1e-6 )
			XCTAssert( rmseΔθσ < 1e-6 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGD() {
		let width: Int = 32 + Int(arc4random_uniform(224))
		let refer: Int = 1 + Int(arc4random_uniform(255))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δv: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.stride, options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width).map(sign), options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.gradient(commandBuffer: commandBuffer, Δv: Δv, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.connect(d: d, φ: φ)
				//$0.jacobian(φ: φ, d: d, j: p)
			}
			commandBuffer.commit()
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: refer)
			let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_pμ) == 0 )
			XCTAssert( la_status(la_pσ) == 0 )
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
			let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_cμ) == 0 )
			XCTAssert( la_status(la_cσ) == 0 )
			
			let la_jμ: la_object_t = la_sum(
				la_scale_with_float(la_outer_product(la_φμ, la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(refer), 1)), 1),
				la_scale_with_float(la_matrix_product(la_d.diagonale, la_pμ), 0)
			)
			let la_jσ: la_object_t = la_matrix_product(
				la_matrix_from_float_buffer(φ.σ.buf.map(recip), la_count_t(width), 1, 1, hint, attr).diagonale,
				la_sum(la_scale_with_float(la_outer_product(la_elementwise_product(la_d, la_elementwise_product(la_φσ, la_φσ)), la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(refer), 1)), 1),
				       la_scale_with_float(la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_pσ), 0)
				)
			)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 < normjσ )
			
			commandBuffer.waitUntilCompleted()
			
			let rmsejμ: Float = la_norm_as_float(la_Δjμ, norm) * rsqrt(Float(width*refer))
			let rmsejσ: Float = la_norm_as_float(la_Δjσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmsejμ < 1e-6 )
			XCTAssert( rmsejσ < 1e-6 )
			
			let la_Δv: la_object_t = la_sum(la_matrix_product(la_Δφμ.diagonale, la_jμ), la_matrix_product(la_Δφσ.diagonale, la_jσ))
			
			XCTAssert( la_status(la_Δv) == 0 )
			
			let normΔv: Float = la_norm_as_float(la_Δv, norm)
			
			XCTAssert( 0 < normΔv )
			
			let la_ΔΔv: la_object_t = la_difference(la_Δv, Δv.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΔv) == 0 )
			
			let rmseΔv: Float = la_norm_as_float(la_ΔΔv, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΔv < 1e-6 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	/*
	func testDeltaGV() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δ: MTLBuffer = device.makeBuffer(array: uniform(count: width * refer), options: [])
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Δ)
			distributor.derivate(commandBuffer: commandBuffer, Δ: Δ, j: j, Δφ: Δφ, count: (rows: width, cols: refer))
			distributor.derivate(commandBuffer: commandBuffer, Δ: Δ, j: j, Δφ: Δφ, count: (rows: width, cols: refer))
			commandBuffer.commit()
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_jμ: la_object_t = j.μ.matrix(rows: width, cols: refer)
			let la_jσ: la_object_t = j.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δμ: la_object_t = la_matrix_product(la_Δφμ.diagonale, la_jμ)
			let la_Δσ: la_object_t = la_matrix_product(la_Δφσ.diagonale, la_jσ)
			
			XCTAssert( la_status(la_Δμ) == 0 )
			XCTAssert( la_status(la_Δσ) == 0 )
			
			let la_ΔΔ: la_object_t = la_difference(la_scale_with_float(la_sum(la_Δμ, la_Δσ), 2), Δ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΔ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Δμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Δσ, norm) )
			
			let rmseΔ: Float = la_norm_as_float(la_ΔΔ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΔ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
		
	}
	func testDeltaGP() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let Δ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Δ)
			distributor.derivate(commandBuffer: commandBuffer, Δ: Δ, j: j, Δφ: Δφ, count: (rows: width, cols: refer))
			distributor.derivate(commandBuffer: commandBuffer, Δ: Δ, j: j, Δφ: Δφ, count: (rows: width, cols: refer))
			commandBuffer.commit()
			
			let la_Δφμ: la_object_t = Δφ.μ.matrix(rows: width, cols: 1)
			let la_Δφσ: la_object_t = Δφ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_Δφμ) == 0 )
			XCTAssert( la_status(la_Δφσ) == 0 )
			
			let la_jμ: la_object_t = j.μ.matrix(rows: width, cols: refer)
			let la_jσ: la_object_t = j.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δμ: la_object_t = la_matrix_product(la_Δφμ.diagonale, la_jμ)
			let la_Δσ: la_object_t = la_matrix_product(la_Δφσ.diagonale, la_jσ)
			
			XCTAssert( la_status(la_Δμ) == 0 )
			XCTAssert( la_status(la_Δσ) == 0 )
			
			let la_ΔΔμ: la_object_t = la_difference(la_scale_with_float(la_Δμ, 2), Δ.μ.matrix(rows: width, cols: refer))
			let la_ΔΔσ: la_object_t = la_difference(la_scale_with_float(la_Δσ, 2), Δ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΔμ) == 0 )
			XCTAssert( la_status(la_ΔΔσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Δμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Δσ, norm) )
			
			let rmseΔμ: Float = la_norm_as_float(la_ΔΔμ, norm) * rsqrt(Float(width*refer))
			let rmseΔσ: Float = la_norm_as_float(la_ΔΔσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΔμ < 1e-5 )
			XCTAssert( rmseΔσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianX() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, x: x, a: a, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_aμ: la_object_t = a.μ.matrix(rows: width, cols: refer)
			let la_aσ: la_object_t = a.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_aμ) == 0 )
			XCTAssert( la_status(la_aσ) == 0 )
			
			let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
			
			XCTAssert( la_status(la_x) == 0 )
			
			let la_Σμ: la_object_t = la_aμ
			let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_aσ, la_aσ), la_x.diagonale)
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianA() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, a: a, x: x, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_aμ: la_object_t = a.μ.matrix(rows: width, cols: refer)
			let la_aσ: la_object_t = a.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_aμ) == 0 )
			XCTAssert( la_status(la_aσ) == 0 )
			
			let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
			
			XCTAssert( la_status(la_x) == 0 )
			
			let la_Σμ: la_object_t = la_outer_product(la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(width), 1), la_x)
			let la_Σσ: la_object_t = la_matrix_product(la_aσ, la_elementwise_product(la_x, la_x).diagonale)
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianB() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let y: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let b: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * width), options: []),
			σ: device.makeBuffer(array: uniform(count: width * width), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, b: b, y: y, g: g, j: j, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_jμ: la_object_t = j.μ.matrix(rows: width, cols: refer)
			let la_jσ: la_object_t = j.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_gμ: la_object_t = g.μ.matrix(rows: width, cols: 1)
			let la_gσ: la_object_t = g.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_gμ) == 0 )
			XCTAssert( la_status(la_gσ) == 0 )
			
			let la_y: la_object_t = y.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_y) == 0 )
			
			let la_bμ: la_object_t = b.μ.matrix(rows: width, cols: width)
			let la_bσ: la_object_t = b.σ.matrix(rows: width, cols: width)
			
			XCTAssert( la_status(la_bμ) == 0 )
			XCTAssert( la_status(la_bσ) == 0 )
			
			let la_Σμ: la_object_t = la_matrix_product(la_bμ, la_matrix_product(la_gμ.diagonale, la_jμ))
			let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_bσ, la_bσ), la_matrix_product(la_elementwise_product(la_y, la_gσ).diagonale, la_jσ))
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianC() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, c: c, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
			let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_cμ) == 0 )
			XCTAssert( la_status(la_cσ) == 0 )
			
			let la_Σμ: la_object_t = la_outer_product(la_matrix_from_splat(la_splat_from_float(1.0, attr), la_count_t(width), 1), la_matrix_from_splat(la_splat_from_float(1.0, attr), la_count_t(refer), 1))
			let la_Σσ: la_object_t = la_outer_product(la_cσ, la_matrix_from_splat(la_splat_from_float(1.0, attr), la_count_t(refer), 1))
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianD() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 1
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, c: c, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_cμ: la_object_t = c.μ.matrix(rows: width, cols: 1)
			let la_cσ: la_object_t = c.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_cμ) == 0 )
			XCTAssert( la_status(la_cσ) == 0 )
			
			let la_Σμ: la_object_t = la_matrix_from_splat(la_splat_from_float(1.0, attr), la_count_t(width), 1)
			let la_Σσ: la_object_t = la_cσ
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianE() {
		let loops: Int = 1 + Int(arc4random_uniform(15))
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			(0..<loops).forEach { (_) in
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, d: d, φ: φ, j: j, count: (rows: width, cols: refer))
			}
			commandBuffer.commit()
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_jμ: la_object_t = j.μ.matrix(rows: width, cols: refer)
			let la_jσ: la_object_t = j.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Σμ: la_object_t = la_matrix_product(la_d.diagonale, la_jμ)
			let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_jσ)
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_ΔΣμ: la_object_t = la_difference(la_scale_with_float(la_Σμ, Float(loops)), Σ.μ.matrix(rows: width, cols: refer))
			let la_ΔΣσ: la_object_t = la_difference(la_scale_with_float(la_Σσ, Float(loops)), Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_ΔΣμ) == 0 )
			XCTAssert( la_status(la_ΔΣσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmseΣμ: Float = la_norm_as_float(la_ΔΣμ, norm) * rsqrt(Float(width*refer))
			let rmseΣσ: Float = la_norm_as_float(la_ΔΣσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmseΣμ < 1e-5 )
			XCTAssert( rmseΣσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianF() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: device.makeBuffer(array: uniform(count: width), options: [])
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: device.makeBuffer(array: uniform(count: width * refer), options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.flush(commandBuffer: commandBuffer, θ: Σ)
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, d: d, φ: φ, j: p, count: (rows: width, cols: refer))
			distributor.jacobian(commandBuffer: commandBuffer, j: j, Σ: Σ, φ: φ, count: (rows: width, cols: refer))
			commandBuffer.commit()
			
			let la_d: la_object_t = d.matrix(rows: width, cols: 1)
			let la_λ: la_object_t = la_matrix_from_float_buffer(φ.σ.buf.map(recip), la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_d) == 0 )
			
			let la_φμ: la_object_t = φ.μ.matrix(rows: width, cols: 1)
			let la_φσ: la_object_t = φ.σ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_φμ) == 0 )
			XCTAssert( la_status(la_φσ) == 0 )
			
			let la_pμ: la_object_t = p.μ.matrix(rows: width, cols: refer)
			let la_pσ: la_object_t = p.σ.matrix(rows: width, cols: refer)
			
			XCTAssert( la_status(la_pμ) == 0 )
			XCTAssert( la_status(la_pσ) == 0 )
			
			let la_Σμ: la_object_t = la_matrix_product(la_d.diagonale, la_pμ)
			let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_elementwise_product(la_d, la_d), la_φσ).diagonale, la_pσ)
			
			XCTAssert( la_status(la_Σμ) == 0 )
			XCTAssert( la_status(la_Σσ) == 0 )
			
			let la_jμ: la_object_t = la_Σμ
			let la_jσ: la_object_t = la_matrix_product(la_λ.diagonale, la_Σσ)
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_Σμ, Σ.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_Σσ, Σ.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 < la_norm_as_float(la_Σμ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Σσ, norm) )
			
			let rmsejμ: Float = la_norm_as_float(la_Δjμ, norm) * rsqrt(Float(width*refer))
			let rmsejσ: Float = la_norm_as_float(la_Δjσ, norm) * rsqrt(Float(width*refer))
			
			XCTAssert( rmsejμ < 1e-5 )
			XCTAssert( rmsejσ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	*/
}
private extension MTLDevice {
	func makeBuffer<T>(array: Array<T>, options: MTLResourceOptions) -> MTLBuffer {
		return makeBuffer(bytes: array, length: array.count * MemoryLayout<T>.stride, options: options)
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
		return Array<T>(UnsafeBufferPointer<T>(start: UnsafePointer<T>(OpaquePointer(cache.contents())), count: cache.length/MemoryLayout<T>.stride))
	}
}
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private extension MTLBuffer {
	func matrix(rows: Int, cols: Int) -> la_object_t {
		XCTAssert(rows*cols*MemoryLayout<Float>.stride<=length)
		return la_matrix_from_float_buffer_nocopy(ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)
	}
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var buf: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: ref, count: count)
	}
	var count: Int {
		return length / MemoryLayout<Float>.stride
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
	
	arc4random_buf(UnsafeMutablePointer<UInt32>(mutating: seeds), count * MemoryLayout<UInt32>.stride)
	
	vDSP_vfltu32(seeds, 1, UnsafeMutablePointer<Float>(mutating: array), 1, vDSP_Length(count))
	vDSP_vsmsa(array, 1, [(β-α)/Float(1<<16)/Float(1<<16)], [α], UnsafeMutablePointer<Float>(mutating: array), 1, vDSP_Length(count))
	
	return array
}
private func sq(_ x: la_object_t) -> la_object_t {
	return la_elementwise_product(x, x)
}
//
private func Φ(_ μ: Float, _ σ: Float) -> Float {
	return Float(fma(erf(0.5.squareRoot()*Double(μ)/Double(σ)), 32767.0/65536.0, 0.5))
}
