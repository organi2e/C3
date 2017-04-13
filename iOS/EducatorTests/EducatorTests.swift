import Metal
import XCTest
import C3
import Optimizer
@testable import Educator
let mnistDB: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTDB.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.499.sqlite")
class EducatorTests: XCTestCase {
	func testWikipedia() {
		do {
			
		} catch {
			
		}
	}
	func testMNIST() {
		do {
			guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let context: Context = try Context(queue: queue,
			                                   storage: storage,
			                                   optimizer: SMORMS3.factory(L2: 1e-6, L1: 0, α: 1e-1))
			if try context.fetch(label: "I", width: 28 * 28).isEmpty {
				print("insert")
				let I: Cell = try context.make(label: "I", width: 28 * 28, distribution: .Gauss, activation: .Identity)
				let H: Cell = try context.make(label: "H", width: 512, distribution: .Gauss, activation: .Binary, input: [I])
				//				let G: Cell = try context.make(label: "G", width: 384, distribution: .Gauss, activation: .Binary, input: [H])
				let F: Cell = try context.make(label: "F", width: 256, distribution: .Gauss, activation: .Binary, input: [H])
				let _:  Cell = try context.make(label: "O", width: 10, distribution: .Gauss, activation: .Binary, input: [F])
				try context.save()
			}
			guard let I: Cell = try context.fetch(label: "I", width: 28 * 28).last else { XCTFail(); return }
			guard let O: Cell = try context.fetch(label: "O", width: 10).last else { XCTFail(); return }
			do {
				let mnist: MNIST = try MNIST(storage: mnistDB)
				if try 0 == mnist.count(group: .train) {
					try mnist.rebuild(group: .train)
					try mnist.save()
				}
				let count: Int = try mnist.count(group: .train)
				let train: Array<Supervised> = try mnist.fetch(group: .train, label: nil)
				
				let batch: Int = 1024
				try stride(from: 0, to: 16384, by: batch).forEach {
					try Array<Void>(repeating: (), count: batch).map{
						try mnist.fetch(group: .train, offset: Int(arc4random_uniform(UInt32(train.count))), limit: 1).first
					}.forEach {
							guard let image: Supervised = $0 else { return }
							O.collect_refresh()
							I.correct_refresh()
							O.target = image.answer
							I.source = image.source
							O.collect()
							I.correct()
					}
					print($0)
					try context.save()
				}
			}
			do {
				let mnist: MNIST = try MNIST(storage: mnistDB)
				if try 0 == mnist.count(group: .t10k) {
					try mnist.rebuild(group: .t10k)
					try mnist.save()
				}
				let batch: Int = 1024
				let count: Int = try mnist.fetch(group: .t10k, offset: 0, limit: batch).map {
					O.collect_refresh()
					I.correct_refresh()
					I.source = $0.source
					O.collect()
					print(O.source, $0.answer)
					defer {
					}
					return O.source == $0.answer
					}.filter{$0}.count
				print(count)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
private func ohv(index: Int, count: Int) -> Array<Float> {
	let array: Array<Float> = Array<Float>(repeating: 0, count: count)
	UnsafeMutablePointer<Float>(mutating: array)[index] = 1.0
	return array
}
