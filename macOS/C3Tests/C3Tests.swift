//
//  C3Tests.swift
//  C3Tests
//
//  Created by Kota Nakano on 2017/04/03.
//
//

import XCTest
import Accelerate
import Metal
import Optimizer
@testable import C3

class C3Tests: XCTestCase {
	let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("C3.sqlite")
	/*
	func testSaveLoad() {
		let label: String = UUID().description
		let width: Int = 1 + Int(arc4random_uniform(15))
		do {
			do {
				let context: Context = try Context(storage: storage)
				let _: Cell = try context.make(label: label, width: width)
				try context.save()
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch()
				print(XS)
				XCTAssert(!XS.isEmpty)
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch(label: label)
				XCTAssert(!XS.isEmpty)
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch(width: width)
				XCTAssert(!XS.isEmpty)
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch(label: label, width: width)
				XCTAssert(XS.count==1)
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch(label: UUID().description, width: width)
				XCTAssert(XS.isEmpty)
			}
			do {
				let context: Context = try Context(storage: storage)
				let XS: [Cell] = try context.fetch(label: label, width: width+1)
				XCTAssert(XS.isEmpty)
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	*/
	func testChain() {
		let storage: URL? = nil//FileManager.default.temporaryDirectory.appendingPathComponent("C3.sqlite")
		do {
			let IS: Array<Array<Float>> = [[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1]]
			let OS: Array<Array<Float>> = [[0,0,0,1], [0,0,1,0], [0,0,1,0], [0,0,0,1]]
			let context: Context = try Context(storage: storage, optimizer: SGD.factory(η: 0.5))
			
			let I: Cell = try context.make(label: "I", width: 4)
			let H: Cell = try context.make(label: "H", width: 256, input: [I])
			let G: Cell = try context.make(label: "G", width: 256, input: [H])
			let O: Cell = try context.make(label: "O", width: 4, input: [G])
			
			/*
			(0..<8).forEach {
				let k: Int = $0 % 4
				print("----")
				
				O.collect_clear()
				I.correct_clear()
				O.target = OS[k]
				I.source = IS[k]
				O.collect()
				I.correct()
				
				print(O.source, O.target)
				print("Y")
				print("χ=", Array(O.state!.buf))
				print("φμ=", Array(O.cache[0].φ.μ.buf))
				print("g=", Array(O.cache[0].g.μ.buf))
				print("Δφ=", Array(O.cache[0].Δφ.μ.buf))
				print("Y-X")
				print("θ=", Array(O.input.first!.θ.μ.buf))
				print("ja=", Array(O.input.first!.ja[0].μ.buf))
				print("Δa=", Array(O.input.first!.Δ.μ.buf))
				print("jx=", Array(O.input.first!.jx[0].μ.buf))
				//print("Δx=", Array(O.input.first!.Δ.μ.buf))
				print("X")
				print("χ=", Array(O.input.first!.input.state!.buf))
				print("φμ=", Array(O.input.first!.input.cache[0].φ.μ.buf))
				print("g=", Array(O.input.first!.input.cache[0].g.μ.buf))
				print("Δφ=", Array(O.input.first!.input.cache[0].Δφ.μ.buf))
				
			}
			*/
			measure {
				(0..<256).forEach {
					let ref: Int = $0 % 4
					O.collect_clear()
					I.correct_clear()
					O.target = OS[ref]
					I.source = IS[ref]
					O.collect()
					I.correct()
				}
			}
			for k in 0..<4 {
				O.collect_clear()
				I.source = IS[k]
				O.collect()
				print(O.source)
			}
			try context.save()
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private extension MTLBuffer {
	func matrix(rows: Int, cols: Int) -> la_object_t {
		XCTAssert(rows * cols * MemoryLayout<Float>.size<=length)
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
private func sq(_ x: la_object_t) -> la_object_t {
	return la_elementwise_product(x, x)
}
