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

private let prefix: String = "MNISTGAN"
private let suffix: String = "v2.3u8x"
private let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
private let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("storage.sqlite")

internal class MNIST {
	let bar: NSProgressIndicator?
	let lab: NSTextField?
	init(progress: NSProgressIndicator?, label: NSTextField?) {
		bar = progress
		lab = label
	}
	func run() {
		guard
			let device: MTLDevice = MTLCreateSystemDefaultDevice(),
			let progress: NSProgressIndicator = bar,
			let label: NSTextField = lab else {
				assertionFailure()
				return
		}
		do {
			let context: Context = try Context(queue: device.makeCommandQueue(),
			                                   storage: storage,
			                                   optimizer: .SMORMS3(L2: 0, L1: 0, α: 1e-1, ε: 0))
			let educator: Educator = try Educator(storage: trainer)
			if try 0 == educator.count(mnist: .train) {
				print("build")
				try autoreleasepool {
					try educator.build(mnist: .train)
					try educator.save()
				}
				educator.reset()
			}
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				try autoreleasepool {
					
					let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 64, distributor: .Degenerate,
					                               activator: .Identity)
					
					let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 256, distributor: .Gauss,
					                               activator: .Binary, input: [I])
					
					let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 768, distributor: .Gauss,
					                               activator: .Binary, input: [H])
					
					let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 28 * 28, distributor: .Degenerate,
					                               activator: .Identity, input: [G])
					
					let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 512, distributor: .Gauss,
					                               activator: .Binary, input: [F])
					
					let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 256, distributor: .Gauss,
					                               activator: .Binary, input: [E])
					
					let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 64, distributor: .Gauss,
					                               activator: .Binary, input: [D])
					
					let _: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 10, distributor: .Gauss,
					                               activator: .Binary, input: [C])
					
					let _: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 1, distributor: .Gauss,
					                               activator: .Binary, input: [C])
					try context.save()
					context.reset()

				}
			}
			let batch: Int = 4000
			let count: Int = try educator.count(mnist: .train)
			try (0..<64).forEach {
				let times: String = String(describing: $0)
				DispatchQueue.main.async {
					label.stringValue = times
				}
				try stride(from: 0, to: count, by: batch).forEach { offset in
					try autoreleasepool {
						guard
							let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
							//								let H: Cell = try context.fetch(label: "\(prefix)H\(suffix)").last,
							//								let G: Cell = try context.fetch(label: "\(prefix)G\(suffix)").last,
							let F: Cell = try context.fetch(label: "\(prefix)F\(suffix)").last,
							let E: Cell = try context.fetch(label: "\(prefix)E\(suffix)").last,
							let D: Cell = try context.fetch(label: "\(prefix)D\(suffix)").last,
							let C: Cell = try context.fetch(label: "\(prefix)C\(suffix)").last,
							let B: Cell = try context.fetch(label: "\(prefix)B\(suffix)").last,
							let A: Cell = try context.fetch(label: "\(prefix)A\(suffix)").last else {
								assertionFailure()
								return
						}
						try CountableRange(uncheckedBounds: (lower: 0, upper: batch)).forEach {
							let ratio: Double = Double($0) / Double(batch)
							DispatchQueue.main.async {
								progress.doubleValue = ratio
							}
							try educator.fetch(mnist: .train, offset: Int(arc4random_uniform(UInt32(count))), limit: 1).forEach {
								
								try A.collect_refresh()
								try B.collect_refresh()
								F.source = $0.source
								try A.collect()
								try B.collect()
								
								try F.correct_refresh()
								A.target = [0]
								B.target = try $0.onehot(count: 10, value: 1)
								try F.correct()
								
								try A.collect_refresh()
								try B.collect_refresh()
								I.source = try $0.onehot(count: 10, value: 1) + Array<Void>(repeating: (), count: 54).map {
									Float(arc4random_uniform(1024))/1024.0
								}
								try A.collect()
								try B.collect()
								
								try I.correct_refresh()
								A.target = [0]
								B.target = try $0.onehot(count: 10, value: 1)
								try I.correct(ignore: [A, B, C, D, E])
								
								try F.correct_refresh()
								A.target = [1]
								try F.correct()
								
							}
						}
						educator.reset()
						/*
						try (0..<10).forEach {
						try educator.fetch(mnist: .train, label: $0, offset: 0, limit: 1).forEach {
						A.collect_refresh()
						D.source = $0.source
						A.collect()
						print(A.source)
						}
						}
						*/
						let output: URL = FileManager.default.temporaryDirectory
						try (0..<60).forEach { index in
							try B.collect_refresh()
							try A.collect_refresh()
							I.source = (0..<10).map {
								Float($0 == index % 10 ? 1 : 0)
								} + Array<Void>(repeating: (), count: 54).map {
									Float(arc4random_uniform(1024))/1024.0
							}
							try B.collect()
							try A.collect()
							print(A.source, B.source)
							try Data(buffer: UnsafeBufferPointer<Float>(start: F.source, count: 28 * 28))
								.write(to: output.appendingPathComponent("img\(index).raw"), options: [])
						}
						context.reset()
						try context.save()
					}
				}
			}
		} catch {
			assertionFailure(String(describing: error))
		}
	}
}
