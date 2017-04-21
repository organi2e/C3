import Metal
import XCTest
import C3
import Optimizer
@testable import Educator
let prefix: String = "GaussBB-"
let suffix: String = "v6"
let corpusDB: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTDB.sqlite")
let mnistDB: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTDB.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.4997.sqlite")
class EducatorTests: XCTestCase {
	/*
	func testWikipedia() {
		do {
			let wikipedia: Wikipedia = try Wikipedia(storage: corpusDB)
			if try wikipedia.count(group: .abstract) == 0 {
				try wikipedia.rebuild(group: .abstract)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	*/
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
					let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 256, distribution: .Gauss, activation: .Binary, input: [I,H])
					let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 128, distribution: .Gauss, activation: .Binary, input: [I,H,G])
					let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width:  64, distribution: .Gauss, activation: .Binary, input: [H,G,F])
					let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width:  10, distribution: .Gauss, activation: .Binary, input: [G,F,E])
					try context.save()
				}
			}
			do {
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L2: 1e-8, L1: 0, α: 0.5))
				let mnist: MNIST = try MNIST(storage: mnistDB)
				if try 0 == mnist.count(group: .train) {
					try mnist.rebuild(group: .train)
					try mnist.save()
				}
				let count: Int = try mnist.count(group: .train)
				try (0..<2).forEach { (_) in
					guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)", width: 28 * 28).last else { XCTFail(); return }
					guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)", width: 10).last else { XCTFail(); return }
					let batch: Int = 1024
					try stride(from: 0, to: count, by: batch).forEach {
						try mnist.fetch(group: .train, offset: $0, limit: batch).forEach {
							O.collect_refresh()
							I.correct_refresh()
							O.target = $0.answer
							I.source = $0.source
							O.collect()
							I.correct()
						}
						print($0)
						mnist.context.reset()
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
				let mnist: MNIST = try MNIST(storage: mnistDB)
				if try 0 == mnist.count(group: .t10k) {
					try mnist.rebuild(group: .t10k)
					try mnist.save()
				}
				let batch: Int = 10002
				let count: Int = try mnist.fetch(group: .t10k, offset: 0, limit: batch).map {
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.source
					O.collect()
					//print(O.source, $0.answer)
					return O.source == $0.answer
					}.filter{$0}.count
				print(count)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
