//
//  C3Tests.swift
//  C3Tests
//
//  Created by Kota Nakano on 2017/04/04.
//
//

import XCTest
import Accelerate
import Metal
import Optimizer
import Adapter
@testable import C3

let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("C3\(UUID().uuidString).sqlite")
//let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("C3\(UUID().uuidString).sqlite")
let IS: Array<Array<Float>> = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
//let GS: Array<Array<Float>> = [[1,1,1,0], [1,1,0,1], [1,1,0,0], [1,0,1,1]]
let OS: Array<Array<Float>> = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
class C3Tests: XCTestCase {
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
		do {
			do {
				let context: Context = try Context(storage: storage)
				let I: Cell = try context.make(label: "I", width: 4)
				let H: Cell = try context.make(label: "H", width: 256, input: [I])
				//let G: Cell = try context.make(label: "G", width: 256, input: [H])
				//let F: Cell = try context.make(label: "F", width: 64, input: [G])
				let _: Cell = try context.make(label: "O", width: 4, input: [H])
				try context.save()
			}
			do {
				let context: Context = try Context(
					storage: storage
					,adapter: (μ: Regular.adapter(), σ: Regular.adapter())
					,optimizer: Adam.factory(α: 1e-1)
				)
				guard let I: Cell = try context.fetch(label: "I").last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "O").last else { XCTFail(); return }
				//measure {
				(0..<1024).forEach {
					let ref: Int = $0 % 4
					O.collect_refresh()
					I.correct_refresh()
					O.target = OS[ref]
					I.source = IS[ref]
					O.collect()
					I.correct()
					print(O.source)
				}
				//}
				try context.save()
			}
			do {
				let context: Context = try Context(
					storage: storage
					//,adapter: (μ: Regular.adapter(), σ: Regular.adapter())
					//,optimizer: SMORMS3.factory(α: 1e-1)
				)
				guard let I: Cell = try context.fetch(label: "I").last else { XCTFail(); return }
				guard let H: Cell = try context.fetch(label: "H").last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "O").last else { XCTFail(); return }
				
				print("gpu")
				for k in 0..<4 {
					O.collect_refresh()
					I.source = IS[k]
					O.collect()
					print(O.source)
				}
				
				print("cpu")
				let (HWμ, HWσ) = context.capture(output: H, input: I)
				let (HCμ, HCσ) = context.capture(cell: H)
				
				let (OWμ, OWσ) = context.capture(output: O, input: H)
				let (OCμ, OCσ) = context.capture(cell: O)
				
				(0..<4).forEach {
					let Xp: LaObjet = make(array: IS[$0], rows: 4, cols: 1)
					
					let Hμ: LaObjet = matrix_product(HWμ, Xp) + HCμ
					let Hv: LaObjet = matrix_product(HWσ*HWσ, Xp*Xp) + HCσ*HCσ
					let Hp: LaObjet = 0.5 + 0.5 * erf(Hμ*rsqrt(2*Hv))
					
					let Oμ: LaObjet = matrix_product(OWμ, Hp) + OCμ
					let Ov: LaObjet = matrix_product(OWσ*OWσ, Hp*Hp) + OCσ*OCσ
					let Op: LaObjet = 0.5 + 0.5 * erf(Oμ*rsqrt(2*Ov))
					
					print(Op.array)
				}
				
				try HWμ.write(to: URL(fileURLWithPath: "/tmp/HWu.raw"))
				try HWσ.write(to: URL(fileURLWithPath: "/tmp/HWs.raw"))
				try HCμ.write(to: URL(fileURLWithPath: "/tmp/HCu.raw"))
				try HCσ.write(to: URL(fileURLWithPath: "/tmp/HCs.raw"))
				try OWμ.write(to: URL(fileURLWithPath: "/tmp/OWu.raw"))
				try OWσ.write(to: URL(fileURLWithPath: "/tmp/OWs.raw"))
				try OCμ.write(to: URL(fileURLWithPath: "/tmp/OCu.raw"))
				try OCσ.write(to: URL(fileURLWithPath: "/tmp/OCs.raw"))
				
			}
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
