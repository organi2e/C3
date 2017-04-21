import Cocoa
import Metal
import XCTest
import Accelerate
import Optimizer
import CoreImage
@testable import C3
@testable import Educator
//let prefix: String = "DegenerateAE-"
let prefix: String = "GAE"
let suffix: String = "v3.9"
let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.sqlite")
class EducatorTests: XCTestCase {
	func testCIFAR10() {
		do {
			let educator: Educator = try Educator(storage: trainer)
//			if try 0 == educator.count(family: .databatch1) {
				try educator.build(family: [.databatch1, .databatch2, .databatch3])
				try educator.save()
//			}
			try educator.fetch(family: .databatch1, limit: 10).enumerated().forEach {
				try CIContext().writeJPEGRepresentation(of: $0.element.ciimage, to: URL(fileURLWithPath: "/tmp/\($0.offset)\($0.element.handle).jpeg"), colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:])
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	/*
	func testMNIST() {
		do {
			guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let context: Context = try Context(queue: queue,
			                                   storage: storage,
			                                   optimizer: SMORMS3.factory(L2: 1e-9, α: 1e-3))
			let educator: Educator = try Educator(storage: trainer)
			if try 0 == educator.count(family: .train) {
				try educator.build(family: .train)
				try educator.save()
				print("build")
			}
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 10, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear))
				let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 64, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [I])
				let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 128, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [H])
				let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 256, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [G])
				let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 512, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [F])
				let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 28 * 28, distribution: .Degenerate, activation: .Identity,
				                               adapters: (.Linear, .Linear), input: [E])
				let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 384, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [D])
				let B: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 64, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [C])
				let _: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 10, distribution: .Gauss, activation: .Binary,
				                               adapters: (.Linear, .Linear), input: [B])
				try context.save()
				context.reset()
			}
			try Array<Void>(repeating: (), count: 1).forEach {
				try autoreleasepool {
					guard
						let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
						let H: Cell = try context.fetch(label: "\(prefix)H\(suffix)").last,
						let G: Cell = try context.fetch(label: "\(prefix)G\(suffix)").last,
						let F: Cell = try context.fetch(label: "\(prefix)F\(suffix)").last,
						let E: Cell = try context.fetch(label: "\(prefix)E\(suffix)").last,
						let D: Cell = try context.fetch(label: "\(prefix)D\(suffix)").last,
						let C: Cell = try context.fetch(label: "\(prefix)C\(suffix)").last,
						let B: Cell = try context.fetch(label: "\(prefix)B\(suffix)").last,
						let A: Cell = try context.fetch(label: "\(prefix)A\(suffix)").last else {
							XCTFail()
							return
					}
					let count: Int = 10000//try educator.count(family: .train)
					let batch: Int = 10000
					try stride(from: 0, to: count, by: batch).forEach {
						print($0)
						try educator.fetch(family: .train, offset: $0, limit: batch).forEach {
							
							I.pliable = false
							H.pliable = false
							G.pliable = false
							F.pliable = false
							E.pliable = false
							D.pliable = false
							C.pliable = true
							B.pliable = true
							A.pliable = true
							
							A.collect_refresh()
							D.correct_refresh()
							A.target = try $0.onehot(count: 10, value: 1)
							D.source = $0.source
							A.collect()
							D.correct()
							
							I.pliable = false
							H.pliable = true
							G.pliable = true
							F.pliable = true
							E.pliable = true
							D.pliable = true
							C.pliable = false
							B.pliable = false
							A.pliable = false
							
							A.collect_refresh()
							I.correct_refresh()
							A.target = try $0.onehot(count: 10, value: 1)
							I.source = try $0.onehot(count: 10, value: 1)
							A.collect()
							I.correct()
							
						}
						try context.save()
						educator.reset()
					}
				}
				context.reset()
			}
			do {
				guard
					let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
					let D: Cell = try context.fetch(label: "\(prefix)D\(suffix)").last,
					let A: Cell = try context.fetch(label: "\(prefix)A\(suffix)").last else {
						XCTFail()
						return
				}
				try (0..<10).forEach {
					try educator.fetch(family: .train, label: $0, offset: 150, limit: 1).forEach {
						A.collect_refresh()
						D.source = $0.source
						A.collect()
						print(A.source)
					}
				}
				let output: URL = URL(fileURLWithPath: "/tmp", isDirectory: true)
				try (0..<60).forEach { index in
					D.collect_refresh()
					I.source = (0..<10).map {
						Float( $0 == index % 10 ? 1 : 0 )
					}
					D.collect()
					try Data(buffer: UnsafeBufferPointer<Float>(start: D.source, count: 28 * 28))
						.write(to: output.appendingPathComponent("img\(index).raw"), options: [])
				}
				print(output)
			}
			/*
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L2: 1e-6, α: 1e-1))
				let mnist: Educator = try Educator(storage: trainer)
				if try 0 == mnist.count(family: .train) {
					try mnist.build(family: .train)
					try mnist.save()
					print("rebuild")
				}
				guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else { XCTFail(); return }
				I.pliable = false
				try Array<Void>(repeating: (), count: 100000).forEach {
					let label: Int = Int(arc4random_uniform(10))
					let limit: Int = 10//try mnist.count(family: .train, label: label, limit: 1)
					guard
						let img0 = try mnist.fetch(family: .train, label: label, offset: Int(arc4random_uniform(UInt32(limit))), limit: 1).first,
						let img1 = try mnist.fetch(family: .train, label: label, offset: Int(arc4random_uniform(UInt32(limit))), limit: 1).first else {
						return
					}
					O.collect_refresh()
					I.correct_refresh()
					O.target = img0.source
					I.source = img1.source
					O.collect()
					I.correct()
				}
				try context.save()
			}
			*/
			/*
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L1: 1e-6, α: 1e-2))
				guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last else { XCTFail(); return }
				guard let H: Cell = try context.fetch(label: "\(prefix)H\(suffix)").last else { XCTFail(); return }
				guard let G: Cell = try context.fetch(label: "\(prefix)G\(suffix)").last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else { XCTFail(); return }
				O.pliable = false
				H.pliable = false
				G.pliable = false
				(0..<16384).forEach { count in
					I.correct_refresh()
					O.collect_refresh()
					O.target = [0,0,0,0,1,0,0,0,0,0]
					O.collect()
					I.correct()
				}
				print(O.source)
//				try Data(bytes: img.contents(), count: img.length).write(to: URL(fileURLWithPath: "/tmp/img.raw")
				try context.save()
				try I.bias.location.write(to: URL(fileURLWithPath: "/tmp/u.raw"))
				try I.bias.scale.write(to: URL(fileURLWithPath: "/tmp/s.raw"))
			}
			*/
			/*
			do {
//				let cicontext: CIContext = CIContext(mtlDevice: device)
				let context: Context = try Context(queue: queue,
				                                   storage: storage)
				guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else { XCTFail(); return }
				let mnist: Educator = try Educator(storage: trainer)
				if try 0 == mnist.count(family: .t10k) {
					try mnist.build(family: .t10k)
					try mnist.save()
				}
				try mnist.fetch(family: .train, label: 0, offset: 3, limit: 1).enumerated().forEach {
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.element.source
					O.collect()
					try Data(bytes: O.source, count: 28 * 28 * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/dec\($0.offset+0).raw"))
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.element.source
					O.collect()
					try Data(bytes: O.source, count: 28 * 28 * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/dec\($0.offset+1).raw"))
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.element.source
					O.collect()
					try Data(bytes: O.source, count: 28 * 28 * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/dec\($0.offset+2).raw"))
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.element.source
					O.collect()
					try Data(bytes: O.source, count: 28 * 28 * MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/dec\($0.offset+3).raw"))
				}
			}
			*/
		} catch {
			XCTFail(String(describing: error))
		}
	}
*/
}
private extension MTLBuffer {
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
}
