//
//  OptimizerTests.swift
//  OptimizerTests
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Accelerate
import Metal
import MetalKit
import Optimizer

import XCTest

class OptimizerTests: XCTestCase {
	
	let count: Int = 256
	
	func uniform(count: Int, range: (α: Float, β: Float) = (α: 0, β: 1)) -> Array<Float> {
		let seed: Array<UInt16> = Array<UInt16>(repeating: 0, count: count)
		let buff: Array<Float> = Array<Float>(repeating: 0, count: count)
		arc4random_buf(UnsafeMutablePointer<UInt16>(mutating: seed), seed.count*MemoryLayout<UInt16>.size)
		vDSP_vfltu16(seed, 1, UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		vDSP_vsmul(buff, 1, [(range.β-range.α)/65536.0], UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		vDSP_vsadd(buff, 1, [range.α], UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		return buff
	}
	func rmse(x: Array<Float>, y: Array<Float>) -> Float {
		var rms: Float = 0
		let val: Array<Float> = Array<Float>(repeating: 0, count: min(x.count, y.count))
		vDSP_vsub(x, 1, y, 1, UnsafeMutablePointer<Float>(mutating: val), 1, vDSP_Length(val.count))
		vDSP_rmsqv(val, 1, &rms, vDSP_Length(val.count))
		return rms
	}
	func uniform(x: MTLBuffer, range: (α: Float, β: Float) = (α: 0, β: 1)) {
		let count: Int = x.length / MemoryLayout<Float>.size
		let seed: Array<UInt16> = Array<UInt16>(repeating: 0, count: count)
		arc4random_buf(UnsafeMutablePointer<UInt16>(mutating: seed), seed.count*MemoryLayout<UInt16>.size)
		vDSP_vfltu16(seed, 1, x.floatPointer, 1, vDSP_Length(count))
		vDSP_vsmul(x.floatPointer, 1, [(range.β-range.α)/65536.0], x.floatPointer, 1, vDSP_Length(count))
		vDSP_vsadd(x.floatPointer, 1, [range.α], x.floatPointer, 1, vDSP_Length(count))
	}
	func prepare(device: MTLDevice, name: String) throws -> MTLComputePipelineState {
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let function: MTLFunction = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
		return try device.makeComputePipelineState(function: function)
	}
	func apply(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState, dydx: MTLBuffer, x: MTLBuffer) {
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(dydx, offset: 0, at: 0)
		encoder.setBuffer(x, offset: 0, at: 1)
		encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	func optimizerTests(factory: (MTLDevice) throws -> (Int) -> Optimizer) {
		do {
			guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
			let optimizer: Optimizer = try factory(device)(count)
			
			let gradientS: MTLComputePipelineState = try prepare(device: device, name: "dydx")
			let gradientN: MTLComputePipelineState = try prepare(device: device, name: "dydx2")
			let θ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let Δθ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			//uniform(x: θ)
			let reset: MTLCommandBuffer = queue.makeCommandBuffer()
			optimizer.reset(commandBuffer: reset)
			reset.commit()
			(0..<1024).forEach { (_) in
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					if drand48() < 0.5 {
						apply(commandBuffer: commandBuffer, pipeline: gradientN, dydx: Δθ, x: θ)
					} else {
						apply(commandBuffer: commandBuffer, pipeline: gradientS, dydx: Δθ, x: θ)
					}
					commandBuffer.commit()
				}
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					optimizer.optimize(commandBuffer: commandBuffer, θ: θ, Δ: Δθ)
					commandBuffer.commit()
				}
			}
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			print(θ.floatArray)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testMomentumAdaDelta() {
		optimizerTests(factory: MomentumAdaDelta.factory())
	}
	func testAdaDelta() {
		optimizerTests(factory: AdaDelta.factory(ρ: 0.9, ε: 1e-6))
	}
	func testAdam() {
		optimizerTests(factory: Adam.factory(α: 1))
	}
	func testMomentum() {
		optimizerTests(factory: Momentum.factory(η: 1e-5, γ: 0.9))
	}
	func testSMORMS3() {
		optimizerTests(factory: SMORMS3.factory())
	}
	func testStochasticGradientDescent() {
		optimizerTests(factory: StochasticGradientDescent.factory(η: 1e-6))
	}
}
extension MTLBuffer {
	var floatPointer: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var floatBuffer: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size)
	}
	var floatArray: Array<Float> {
		return Array<Float>(UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size))
	}
	var mse: Float {
		var mse: Float = 0
		vDSP_rmsqv(UnsafePointer<Float>(OpaquePointer(contents())), 1, &mse, vDSP_Length(length/MemoryLayout<Float>.size))
		return mse
	}
}
