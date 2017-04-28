import Metal
import XCTest
import C3
import Optimizer
@testable import Educator
let prefix: String = "GaussBB-"
let suffix: String = "v8"
let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.49991.sqlite")
class EducatorTests: XCTestCase {
	func testMNIST() {
		do {
			guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
			let queue: MTLCommandQueue = device.makeCommandQueue()
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage)
				if try context.fetch(label: "\(prefix)I\(suffix)", width: 28 * 28).isEmpty {
					print("insert")
					let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 28 * 28, distribution: .Gauss, activation: .Identity)
					let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 512, distribution: .Gauss, activation: .Binary, input: [I])
					let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 256, distribution: .Gauss, activation: .Binary, input: [H])
//					let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width:  64, distribution: .Gauss, activation: .Binary, input: [G])
					let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width:  10, distribution: .Gauss, activation: .Binary, input: [G])
					try context.save()
				}
			}
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: .SMORMS3(L2: 1e-6, L1: 0, α: 1e-3, ε: 0))
				let mnist: Educator = try Educator(storage: trainer)
				if try 0 == mnist.count(mnist: .train) {
					try mnist.build(mnist: .train)
					try mnist.save()
				}
				let count: Int = try mnist.count(mnist: .train)
				try (0..<1).forEach { (_) in
					guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)", width: 28 * 28).last else { XCTFail(); return }
					guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)", width: 10).last else { XCTFail(); return }
					let batch: Int = 1024
					try stride(from: 0, to: count, by: batch).forEach {
						print($0)
						try mnist.fetch(mnist: .train, offset: $0, limit: batch).forEach {
							O.collect_refresh()
							I.correct_refresh()
							O.target = try $0.onehot(count: 10, value: 1)
							I.source = $0.source
							O.collect()
							I.correct()
						}
						try mnist.save()
						try context.save()
					}
					try context.save()
					context.reset()
				}
			}
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage)
				guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)", width: 28 * 28).last else { XCTFail(); return }
				guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)", width: 10).last else { XCTFail(); return }
				let mnist: Educator = try Educator(storage: trainer)
				if try 0 == mnist.count(mnist: .t10k) {
					try mnist.build(mnist: .t10k)
					try mnist.save()
				}
				let batch: Int = 1024
				let count: Int = try mnist.fetch(mnist: .t10k, limit: batch).map {
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.source
					O.collect()
					print(O.source, $0.handle)
					return try O.source == $0.onehot(count: 10, value: 1)
				}.filter{$0}.count
				print(count)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
