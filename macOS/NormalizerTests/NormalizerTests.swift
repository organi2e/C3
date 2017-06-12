//
//  NormalizerTests.swift
//  NormalizerTests
//
//  Created by Kota Nakano on 2017/06/09.
//
//

import Metal
import XCTest
@testable import Normalizer

class NormalizerTests: XCTestCase {
    func testExample() {
		let count: Int = 32
		do {
			guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else {
				throw NSError(domain: "", code: 0, userInfo: nil)
			}
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let normalizer: Normalizer = try Normalizer(device: device, count: count, γ: 0.995)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			normalizer.reset(commandBuffer: commandBuffer)
			commandBuffer.commit()
			(0..<4096).forEach {
				let source: MTLBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				source.ref[0] = fma(Float($0 % 2), 4.0, -3.0)
				normalizer.connect(commandBuffer: commandBuffer, source: source)
				commandBuffer.commit()
			}
			stride(from: Float(0), to: 10, by: 1).forEach {
				let source: MTLBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
				let target: MTLBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				source.ref[0] = fma(fmod($0, 2.0), 4.0, -3.0)
				source.ref[1] = $0
				normalizer.collect(commandBuffer: commandBuffer, target: target, source: source)
				commandBuffer.addCompletedHandler { (_) in
					print(source.ref[0], source.ref[1], target.ref[0], target.ref[1])
				}
				commandBuffer.commit()
			}
		} catch {
			XCTFail(String(describing: error))
		}
    }
}
private extension MTLBuffer {
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
}
