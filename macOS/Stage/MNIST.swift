//
//  MNIST.swift
//  macOS
//
//  Created by user00 on 4/26/17.
//
//

import Cocoa
import Metal
import Accelerate
import C3
import Educator

private let trainerURL: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
private let storageURL: URL = FileManager.default.temporaryDirectory.appendingPathComponent("storage.sqlite")

internal class MNIST {
	let context: Context
	let trainer: Educator
	let bar: NSProgressIndicator
	let lab: NSTextField
	init(progress: NSProgressIndicator?, label: NSTextField?) throws {
		guard
			let device: MTLDevice = MTLCreateSystemDefaultDevice(),
			let progress: NSProgressIndicator = progress,
			let label: NSTextField = label else {
			throw NSError(domain: #function, code: #line, userInfo: nil)
		}
		context = try Context(queue: device.makeCommandQueue(),
		                      storage: storageURL,
		                      optimizer: .SMORMS3(L2: 0, L1: 0, α: 1e-3, ε: 0))
		trainer = try Educator(storage: trainerURL)
		bar = progress
		lab = label
		if try 0 == trainer.count(mnist: .train) {
			print("build train")
			try autoreleasepool {
				try trainer.build(mnist: .train)
				try trainer.save()
			}
			trainer.reset()
		}
		if try 0 == trainer.count(mnist: .t10k) {
			print("build t10k")
			try autoreleasepool {
				try trainer.build(mnist: .t10k)
				try trainer.save()
			}
			trainer.reset()
		}
	}
	func semisupervised() {
		let prefix: String = "MNISTSemisupervised"
		let suffix: String = "v1.1"
		do {
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 28 * 28, distributor: .Gauss, regularizer: 0e-0, activator: .Binary)
				let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 512, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [I])
				let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 256, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [H])
				let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 64, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [G])
				let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width: 10, distributor: .Gauss, regularizer: 0e-0, activator: .Binary, input: [F])
				try context.save()
			}
			let batch: Int = 5000
			let count: Int = try trainer.count(mnist: .train)
			try (0..<64).forEach {
				let times: String = String(describing: $0)
				DispatchQueue.main.async {
					self.lab.stringValue = times
				}
				try stride(from: 0, to: count, by: batch).forEach { offset in
					try autoreleasepool {
						guard
							let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
							let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else {
								assertionFailure()
								return
						}
						try trainer.fetch(mnist: .train, offset: offset, limit: batch).enumerated().forEach {
							let ratio: Double = Double($0) / Double(batch)
							DispatchQueue.main.async {
								self.bar.doubleValue = ratio
							}
							
							try O.collect_refresh()
							I.source = $1.source
							try O.collect()
							
							try I.correct_refresh()
							if arc4random_uniform(1000) < 1 {
								O.target = try $1.onehot(count: 10, value: 1)
							}
							try I.correct()
							
						}
						try context.save()
						try (0..<10).forEach {
							try trainer.fetch(mnist: .t10k,
							                  label: $0,
							                  offset: Int(arc4random_uniform(UInt32(trainer.count(mnist: .t10k, label: $0)))),
							                  limit: 1).forEach {
																	try O.collect_refresh()
																	I.source = $0.source
																	try O.collect()
																	try print(O.source, $0.onehot(count: 10, value: 1))
							}
						}
					}
				}
			}
		} catch {
			assertionFailure(String(describing: error))
		}
	}
	func supervised() {
		let prefix: String = "MNISTSupervised"
		let suffix: String = "v1.1"
		do {
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 28 * 28, distributor: .Gauss, regularizer: 0e-0, activator: .Binary)
				let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 512, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [I])
				let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 256, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [H])
				let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 64, distributor: .Gauss, regularizer: 1e-3, activator: .Binary, input: [G])
				let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width: 10, distributor: .Gauss, regularizer: 0e-0, activator: .Binary, input: [F])
				try context.save()
			}
			let batch: Int = 5000
			let count: Int = try trainer.count(mnist: .train)
			try (0..<64).forEach {
				let times: String = String(describing: $0)
				DispatchQueue.main.async {
					self.lab.stringValue = times
				}
				try stride(from: 0, to: count, by: batch).forEach { offset in
					try autoreleasepool {
						guard
							let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
							let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else {
								assertionFailure()
								return
						}
						try trainer.fetch(mnist: .train, offset: offset, limit: batch).enumerated().forEach {
							let ratio: Double = Double($0) / Double(batch)
							DispatchQueue.main.async {
								self.bar.doubleValue = ratio
							}
							
							try O.collect_refresh()
							I.source = $1.source
							try O.collect()
							
							try I.correct_refresh()
							O.target = try $1.onehot(count: 10, value: 1)
							try I.correct()
							
						}
						try context.save()
						try (0..<10).forEach {
							try trainer.fetch(mnist: .t10k,
							                  label: $0,
							                  offset: Int(arc4random_uniform(UInt32(trainer.count(mnist: .t10k, label: $0)))),
							                  limit: 1).forEach {
								try O.collect_refresh()
								I.source = $0.source
								try O.collect()
								try print(O.source, $0.onehot(count: 10, value: 1))
							}
						}
					}
				}
			}
		} catch {
			assertionFailure(String(describing: error))
		}
	}
	func gan() {
		let prefix: String = "MNISTGan"
		let suffix: String = "v1.1"
		do {
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 64, distributor: .Gauss, regularizer: 0, activator: .Binary)
				let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [I])
				let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [H])
				let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [G])
				let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 784, distributor: .Gauss, regularizer: 0, activator: .Identity, input: [F])
				let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [E])
				let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [D])
				let B: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 1024, distributor: .Gauss, regularizer: 1e-2, activator: .Binary, input: [C])
				let _: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 64, distributor: .Gauss, regularizer: 0, activator: .Binary, input: [B])
				try context.save()
			}
			let batch: Int = 3000
			let count: Int = try trainer.count(mnist: .train)
			try (0..<64).forEach {
				let times: String = String(describing: $0)
				DispatchQueue.main.async {
					self.lab.stringValue = times
				}
				try stride(from: 0, to: count, by: batch).forEach { offset in
					try autoreleasepool {
						guard
							let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
							let _: Cell = try context.fetch(label: "\(prefix)H\(suffix)").last,
							let _: Cell = try context.fetch(label: "\(prefix)G\(suffix)").last,
							let _: Cell = try context.fetch(label: "\(prefix)F\(suffix)").last,
							let E: Cell = try context.fetch(label: "\(prefix)E\(suffix)").last,
							let D: Cell = try context.fetch(label: "\(prefix)D\(suffix)").last,
							let C: Cell = try context.fetch(label: "\(prefix)C\(suffix)").last,
							let B: Cell = try context.fetch(label: "\(prefix)B\(suffix)").last,
							let A: Cell = try context.fetch(label: "\(prefix)A\(suffix)").last else {
								assertionFailure()
								return
						}
						try trainer.fetch(mnist: .train, offset: offset, limit: batch).enumerated().forEach {
							let ratio: Double = Double($0) / Double(batch)
							DispatchQueue.main.async {
								self.bar.doubleValue = ratio
							}
							
							try A.collect_refresh()
							E.source = $1.source
							try A.collect()
							
							try E.correct_refresh()
							A.target = try $1.onehot(count: 10, value: 1) + Array<Float>(repeating: 0, count: 54)
							try E.correct()
							
							try A.collect_refresh()
							I.source = try $1.onehot(count: 10, value: 1) + uniform(count: 54)
							try A.collect()
							
							try I.correct_refresh()
							A.target = try $1.onehot(count: 10, value: 1) + Array<Float>(repeating: 0, count: 54)
							try I.correct(ignore: [A, B, C, D])
							
							try E.correct_refresh()
							C.target = try $1.onehot(count: 10, value: 1) + Array<Float>(repeating: 1, count: 54)
							try E.correct()
							
						}
						try context.save()
						print("D")
						try (0..<10).forEach {
							try trainer.fetch(mnist: .train,
							                  label: $0,
							                  offset: Int(arc4random_uniform(UInt32(trainer.count(mnist: .train, label: $0)))),
							                  limit: 1).forEach {
								try A.collect_refresh()
								E.source = $0.source
								try A.collect()
								try print(A.source, $0.onehot(count: 10, value: 1))
							}
						}
						print("G")
						try (0..<60).forEach {
							try E.collect_refresh()
							I.source = try ($0%10).onehot(count: 10, value: 1) + uniform(count: 54)
							try E.collect()
//							try print(C.source, $0.onehot(count: 10, value: 1))
							let buf: Array<Float> = E.source
							try Data(bytes: buf, count: buf.count * MemoryLayout<Float>.stride)
								.write(to: FileManager.default.temporaryDirectory.appendingPathComponent(String($0)).appendingPathExtension("raw"))
						}
					}
				}
			}
		} catch {
			assertionFailure(String(describing: error))
		}
	}
}
private func uniform(count: Int) -> Array<Float> {
	let result: Array<Float> = Array<Float>(repeating: 0, count: count)
	let seed: Array<UInt8> = Array<UInt8>(repeating: 0, count: count)
	arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seed), seed.count * MemoryLayout<UInt8>.stride)
	vDSP_vfltu8(seed, 1, UnsafeMutablePointer<Float>(mutating: result), 1, vDSP_Length(count))
	return result
}
