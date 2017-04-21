//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import Cocoa
import Metal

import C3
import Optimizer
import Educator

let prefix: String = "GAE"
let suffix: String = "v1.3"
let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("stage.sqlite")
class ViewController: NSViewController {

	@IBOutlet var progress: NSProgressIndicator?
	@IBOutlet var label: NSTextField?
	
	override func viewDidLoad() {
		super.viewDidLoad()
		guard
			let device: MTLDevice = MTLCreateSystemDefaultDevice(),
			let progress: NSProgressIndicator = progress,
			let label: NSTextField = label else {
				assertionFailure()
				return
		}
		Thread {
			do {
				let context: Context = try Context(queue: device.makeCommandQueue(),
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L2: 5e-8, L1: 0, α: 5e-4))
				let educator: Educator = try Educator(storage: trainer)
				if try 0 == educator.count(family: .train) {
					print("build")
					try autoreleasepool {
						try educator.build(family: .train)
						try educator.save()
					}
					educator.reset()
				}
				if try 0 == context.count(label: "\(prefix)I\(suffix)") {
					print("insert")
					try autoreleasepool {
						let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 16,
						                               distribution: .Gauss, activation: .Binary)
						let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 64,
						                               distribution: .Gauss, activation: .Binary, input: [I])
						let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 128,
						                               distribution: .Gauss, activation: .Binary, input: [H])
						let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 256,
						                               distribution: .Gauss, activation: .Binary, input: [G])
						let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 512,
						                               distribution: .Gauss, activation: .Binary, input: [F])
						let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 28 * 28,
						                               distribution: .Degenerate, activation: .Identity, input: [E])
						let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 384,
						                               distribution: .Gauss, activation: .Binary, input: [D])
						let B: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 64,
						                               distribution: .Gauss, activation: .Binary, input: [C])
						let _: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 16,
						                               distribution: .Degenerate, activation: .Binary, input: [B])
						try context.save()
						context.reset()
					}
				}
				let batch: Int = 10000
				let count: Int = try educator.count(family: .train)
				try (0..<1000).forEach {
					let times: String = String(describing: $0)
					DispatchQueue.main.async {
						label.stringValue = times
					}
					try stride(from: 0, to: count, by: batch).forEach { offset in
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
									assertionFailure()
									return
							}
							try educator.fetch(family: .train, offset: offset, limit: batch).enumerated().forEach {
								let ratio: Double = Double($0.offset) / Double(batch)
								DispatchQueue.main.async {
									progress.doubleValue = ratio
								}
								//G
								A.collect_refresh()
								D.source = $0.element.source
								A.collect()
								
								I.pliable = false
								H.pliable = false
								G.pliable = false
								F.pliable = false
								E.pliable = false
								D.pliable = false
								C.pliable = true
								B.pliable = true
								A.pliable = true
								
								D.correct_refresh()
								A.target = try $0.element.onehot(count: 10, value: 1) + [0]
								D.correct()
								
								//
								A.collect_refresh()
								I.source = try $0.element.onehot(count: 10, value: 1)
								A.collect()
								
								//N
								I.pliable = false
								H.pliable = false
								G.pliable = false
								F.pliable = false
								E.pliable = false
								D.pliable = false
								C.pliable = true
								B.pliable = true
								A.pliable = true
								
								D.correct_refresh()
								A.target = try $0.element.onehot(count: 10, value: 1) + [1]
								D.correct()
								
								//A
								I.pliable = false
								H.pliable = true
								G.pliable = true
								F.pliable = true
								E.pliable = true
								D.pliable = true
								C.pliable = false
								B.pliable = false
								A.pliable = false
								
								I.correct_refresh()
								A.target = try $0.element.onehot(count: 10, value: 1) + [0]
								I.correct()
								
							}
							try context.save()
							try (0..<10).forEach {
								try educator.fetch(family: .train, label: $0, offset: 0, limit: 1).forEach {
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
							context.reset()
							educator.reset()
						}
					}
				}
			} catch {
				assertionFailure(String(describing: error))
			}
		}.start()
	}

	override var representedObject: Any? {
		didSet {
		// Update the view, if already loaded.
		}
	}
	

}
extension Data {
	func split(cursor: Int) -> (Data, Data) {
		let m: Data.Index = startIndex.advanced(by: count)
		return (subdata(in: startIndex..<m), subdata(in: m..<endIndex))
	}
	func toArray<T>() -> Array<T> {
		return withUnsafeBytes { Array<T>(UnsafeBufferPointer<T>(start: $0, count: count / MemoryLayout<T>.size)) }
	}
}
extension Array {
	func chunk(count: Int) -> [[ Element ]] {
		return stride(from: 0, to: count, by: count).map {
			Array(self[$0..<$0.advanced(by: count)])
		}
	}
}
