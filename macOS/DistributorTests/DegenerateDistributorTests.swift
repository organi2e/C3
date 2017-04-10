//
//  DegenerateDistributorTests.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/10.
//
//
import Metal
import Accelerate
import XCTest
import simd
@testable import Distributor
class DegenerateDistributorTests: XCTestCase {
	func testAF() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width*refer), options: []),
			σ: discard
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		
		let la_w: la_object_t = w.μ.matrix(rows: width, cols: refer)
		let la_c: la_object_t = c.μ.matrix(rows: width, cols: 1)
		let la_p: la_object_t = p.μ.matrix(rows: width, cols: 1)
		let la_d: la_object_t = d.matrix(rows: width, cols: 1)
		let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let χ: MTLBuffer = device.makeBuffer(length: width*MemoryLayout<Float>.size, options: [])
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, f: χ, g: g, φ: φ, count: width) {
				$0.collect(w: w, x: x, count: refer)
				$0.collect(c: c)
				$0.collect(d: d, φ: p)
			}
			commandBuffer.commit()
			
			let la_φ: la_object_t = [
				la_matrix_product(la_w, la_x),
				la_elementwise_product(la_d, la_p)
			].reduce(la_c, la_sum)
			let la_χ: la_object_t = la_matrix_from_float_buffer(la_φ.array.map{step($0, edge: 0.0)}, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_φ) == 0 )
			XCTAssert( la_status(la_χ) == 0 )
			
			let la_Δφ: la_object_t = la_difference(la_φ, φ.μ.matrix(rows: width, cols: 1))
			let la_Δχ: la_object_t = la_difference(la_χ, χ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δφ) == 0 )
			XCTAssert( la_status(la_Δχ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 == la_norm_as_float(g.σ.matrix(rows: width, cols: 1), norm) )
			XCTAssert( 0 == la_norm_as_float(φ.σ.matrix(rows: width, cols: 1), norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_φ, norm) )
			XCTAssert( 0 < la_norm_as_float(la_χ, norm) )
			
			let rmseφ: Float = la_norm_as_float(la_Δφ, norm) * rsqrt(Float(width))
			let rmseχ: Float = la_norm_as_float(la_Δχ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseφ < 1e-5 )
			XCTAssert( rmseχ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testAV() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width*refer), options: []),
			σ: discard
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		
		let la_w: la_object_t = w.μ.matrix(rows: width, cols: refer)
		let la_c: la_object_t = c.μ.matrix(rows: width, cols: 1)
		let la_p: la_object_t = p.μ.matrix(rows: width, cols: 1)
		let la_d: la_object_t = d.matrix(rows: width, cols: 1)
		let la_x: la_object_t = x.matrix(rows: refer, cols: 1)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let χ: MTLBuffer = device.makeBuffer(length: width*MemoryLayout<Float>.size, options: [])
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, v: χ, g: g, φ: φ, count: width) {
				$0.collect(w: w, x: x, count: refer)
				$0.collect(c: c)
				$0.collect(d: d, φ: p)
			}
			commandBuffer.commit()
			
			let la_φ: la_object_t = [
				la_matrix_product(la_w, la_x),
				la_elementwise_product(la_d, la_p),
				la_c
			].reduce(la_splat_from_float(0, attr), la_sum)
			
			XCTAssert( la_status(la_φ) == 0 )
			
			let la_Δφ: la_object_t = la_difference(la_φ, φ.μ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δφ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 == la_norm_as_float(g.σ.matrix(rows: width, cols: 1), norm) )
			XCTAssert( 0 == la_norm_as_float(φ.σ.matrix(rows: width, cols: 1), norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_φ, norm) )
			
			let rmseφ: Float = la_norm_as_float(la_Δφ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseφ < 1e-5 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testDF() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: 0, β: 1), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: -1, β: 1), options: []),
			σ: discard
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let χ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let f: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let ϝ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		do {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			let distributor: Distributor = try DegenerateDistributor(device: device)
			distributor.activate(commandBuffer: commandBuffer, f: f, g: g, φ: φ, count: width) {
				$0.collect(c: c)
			}
			distributor.activate(commandBuffer: commandBuffer, Δφ: Δφ, f: f, g: g, φ: φ, count: width) {
				$0.correct(χ: χ, ϝ: ϝ)
			}
			commandBuffer.commit()
			
			let buf_p: Array<Float> = c.μ.buf.map { 1.0 / ( 1.0 + exp( -$0 ) ) }
			let buf_g: Array<Float> = buf_p.map { $0 * ( 1.0 - $0 ) }
			let la_g: la_object_t = la_matrix_from_float_buffer(buf_g, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_g) == 0 )
			
			let la_Δg: la_object_t = la_difference(la_g, g.μ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δg) == 0 )
			
			let la_χ: la_object_t = χ.matrix(rows: width, cols: 1)
			let la_ϝ: la_object_t = ϝ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_χ) == 0 )
			XCTAssert( la_status(la_ϝ) == 0 )
			
			let la_Δ: la_object_t = la_matrix_from_float_buffer([
				la_difference(la_χ, la_ϝ)
			].reduce(la_splat_from_float(0, attr), la_sum).array.map(sign), la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_Δ) == 0 )
			
			let la_Δφ: la_object_t = la_matrix_product(la_g.diagonale, la_Δ)
			
			XCTAssert( la_status(la_Δφ) == 0 )
			
			let la_ΔΔφ: la_object_t = la_difference(la_Δφ, Δφ.μ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_ΔΔφ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 == la_norm_as_float(g.σ.matrix(rows: width, cols: 1), norm) )
			XCTAssert( 0 == la_norm_as_float(Δφ.σ.matrix(rows: width, cols: 1), norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_g, norm) )
			XCTAssert( 0 < la_norm_as_float(la_Δφ, norm) )
			
			let rmseg: Float = la_norm_as_float(la_Δg, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseg < 1e-6 )
			
			let rmseΔφ: Float = la_norm_as_float(la_ΔΔφ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseΔφ < 1e-6 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testDV() {
		let width: Int = 16 + Int(arc4random_uniform(240))
		let refer: Int = 16 + Int(arc4random_uniform(240))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: 0, β: 1), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width, α: -1, β: 1), options: []),
			σ: discard
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let χ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let v: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let ϝ: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		do {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			let distributor: Distributor = try DegenerateDistributor(device: device)
			distributor.activate(commandBuffer: commandBuffer, v: v, g: g, φ: φ, count: width) {
				$0.collect(c: c)
			}
			distributor.activate(commandBuffer: commandBuffer, Δφ: Δφ, v: v, g: g, φ: φ, count: width) {
				$0.correct(χ: χ, ϝ: ϝ)
			}
			commandBuffer.commit()
			
			let la_g: la_object_t = la_matrix_from_splat(la_splat_from_float(1, attr), la_count_t(width), la_count_t(1))
			
			XCTAssert( la_status(la_g) == 0 )
			
			let la_Δg: la_object_t = la_difference(la_g, g.μ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_Δg) == 0 )
			
			let la_χ: la_object_t = χ.matrix(rows: width, cols: 1)
			let la_ϝ: la_object_t = ϝ.matrix(rows: width, cols: 1)
			
			XCTAssert( la_status(la_χ) == 0 )
			XCTAssert( la_status(la_ϝ) == 0 )
			
			let la_Δ: la_object_t = la_matrix_from_float_buffer([
				la_difference(la_χ, la_ϝ)
			].reduce(la_splat_from_float(0, attr), la_sum).array, la_count_t(width), 1, 1, hint, attr)
			
			XCTAssert( la_status(la_Δ) == 0 )
			
			let la_Δφ: la_object_t = la_Δ
			
			XCTAssert( la_status(la_Δφ) == 0 )
			
			let la_ΔΔφ: la_object_t = la_difference(la_Δφ, Δφ.μ.matrix(rows: width, cols: 1))
			
			XCTAssert( la_status(la_ΔΔφ) == 0 )
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert( 0 == la_norm_as_float(g.σ.matrix(rows: width, cols: 1), norm) )
			XCTAssert( 0 == la_norm_as_float(Δφ.σ.matrix(rows: width, cols: 1), norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_g, norm) )
			
			XCTAssert( 0 < la_norm_as_float(la_Δφ, norm) )
			
			let rmseg: Float = la_norm_as_float(la_Δg, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseg < 1e-6 )
			
			let rmseΔφ: Float = la_norm_as_float(la_ΔΔφ, norm) * rsqrt(Float(width))
			
			XCTAssert( rmseΔφ < 1e-6 )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGX() {
		let width: Int = 32 + Int(arc4random_uniform(224))
		let refer: Int = 32 + Int(arc4random_uniform(224))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let Δx: MTLBuffer = device.makeBuffer(length: refer * MemoryLayout<Float>.size, options: [])
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.derivate(commandBuffer: commandBuffer, Δx: Δx, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.jacobian(x: x, a: a)
				$0.jacobian(φ: φ, d: d, j: p)
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
			let la_jσ: la_object_t = la_matrix_from_splat(la_splat_from_float(0, attr), la_count_t(width), la_count_t(refer))
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 == normjσ )
			
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
		let width: Int = 32// + Int(arc4random_uniform(224))
		let refer: Int = 4// + Int(arc4random_uniform(224))
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let Δθ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let x: MTLBuffer = device.makeBuffer(array: uniform(count: refer), options: [])
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.derivate(commandBuffer: commandBuffer, Δθ: Δθ, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.jacobian(a: a, x: x)
				$0.jacobian(φ: φ, d: d, j: p)
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
			let la_jσ: la_object_t = la_matrix_from_splat(la_splat_from_float(0, attr), la_count_t(width), la_count_t(refer))
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 == normjσ )
			
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
			XCTAssert( 0 == normΔθσ )
			
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
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let Δθ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.derivate(commandBuffer: commandBuffer, Δθ: Δθ, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.jacobian(c: c)
				$0.jacobian(φ: φ, d: d, j: p)
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
			let la_jσ: la_object_t = la_matrix_from_splat(la_splat_from_float(0, attr), la_count_t(width), la_count_t(refer))
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 == normjσ )
			
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
			XCTAssert( 0 == normΔθσ )
			
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
		let discard: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let Δv: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: [])
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: []),
			σ: discard
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let d: MTLBuffer = device.makeBuffer(array: uniform(count: width), options: [])
		let p: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width * refer), options: []),
			σ: discard
		)
		let Δφ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		let φ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(array: uniform(count: width), options: []),
			σ: discard
		)
		do {
			let distributor: Distributor = try DegenerateDistributor(device: device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.derivate(commandBuffer: commandBuffer, Δv: Δv, j: j, Δφ: Δφ, φ: φ, count: (rows: width, cols: refer)) {
				$0.jacobian(d: d, φ: φ)
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
			let la_jσ: la_object_t = la_matrix_from_splat(la_splat_from_float(0, attr), la_count_t(width), la_count_t(refer))
			
			XCTAssert( la_status(la_jμ) == 0 )
			XCTAssert( la_status(la_jσ) == 0 )
			
			let la_Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: width, cols: refer))
			let la_Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: width, cols: refer))
			
			XCTAssert( la_status(la_Δjμ) == 0 )
			XCTAssert( la_status(la_Δjσ) == 0 )
			
			let normjμ: Float = la_norm_as_float(la_jμ, norm)
			let normjσ: Float = la_norm_as_float(la_jσ, norm)
			
			XCTAssert( 0 < normjμ )
			XCTAssert( 0 == normjσ )
			
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
		return la_matrix_from_float_buffer_nocopy(ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)
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
