import Metal
import XCTest
import C3
import Optimizer
@testable import Educator
let mnistDB: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTDB.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.495.sqlite")
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
				let G: Cell = try context.make(label: "G", width: 384, distribution: .Gauss, activation: .Binary, input: [H])
				let F: Cell = try context.make(label: "F", width: 256, distribution: .Gauss, activation: .Binary, input: [G])
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
				let train: Array<Supervised> = try mnist.fetch(group: .train, label: nil)
				for k in 0..<64 {
					let index: Int = Int(arc4random_uniform(UInt32(train.count)))
					O.collect_refresh()
					I.correct_refresh()
					O.target = train[index].answer
					I.source = train[index].source
					O.collect()
					I.correct()
					//if k % 64 == 62 {
					//	try context.save()
					//	context.refreshAllObjects()
					//}
				}
				try context.save()
			}
			do {
				let mnist: MNIST = try MNIST(storage: mnistDB)
				if try 0 == mnist.count(group: .t10k) {
					try mnist.rebuild(group: .t10k)
					try mnist.save()
				}
				var count: Int = 0
				let batch: Int = 32
				try stride(from: 0, to: 1024, by: batch).forEach {
					let image = try mnist.fetch(group: .t10k, offset: $0, limit: batch)
					image.forEach { (_) in
						O.collect_refresh()
						I.correct_refresh()
						I.source = image[0].source
						O.collect()
						let match: Bool = O.source == image[0].answer
						//print(x, $0.answer, match)
						count = count + ( match ? 1 : 0 )
					}
					print($0, "reset")
//					mnist.context.reset()
				}
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
