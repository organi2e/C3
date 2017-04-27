//
//  PTB.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/24.
//
//
import Cocoa
import Metal
import Accelerate
import Optimizer
import C3
import Educator

private let prefix: String = "ASCIImod"
private let suffix: String = "v0.3"
private let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
private let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("storage.sqlite")

internal class PTB {
	let bar: NSProgressIndicator?
	let lab: NSTextField?
	init(progress: NSProgressIndicator?, label: NSTextField?) {
		bar = progress
		lab = label
	}
	internal func run() {
		guard
			let device: MTLDevice = MTLCreateSystemDefaultDevice(),
			let progress: NSProgressIndicator = bar,
			let label: NSTextField = lab else {
				assertionFailure()
				return
		}
		do {
			let educator: Educator = try Educator(storage: trainer)
			if try 0 == educator.count(ptb: .test) {
				try educator.build(ptb: .test)
				try educator.save()
				print("build")
			}
			guard let string: ContiguousArray<CChar> = try educator.fetch(ptb: .test).first?.text.components(separatedBy: CharacterSet.whitespacesAndNewlines).joined(separator: " ").utf8CString else {
				assertionFailure()
				return
			}
			educator.reset()
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let context: Context = try Context(queue: queue,
			                                   storage: storage,
			                                   optimizer: SMORMS3.factory(L2: 1e-6, α: 1e-3))
			if try 0 == context.count(label: "\(prefix)I\(suffix)") {
				print("insert")
				try autoreleasepool {
					let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 256, distribution: .Gauss, activation: .Binary,
					                               adapters: (.Linear, .Softplus))
					let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 384, distribution: .Gauss, activation: .Binary,
					                               adapters: (.Linear, .Softplus), input: [I], decay: true, recurrent: [-1])
					let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 512, distribution: .Gauss, activation: .Binary,
					                               adapters: (.Linear, .Softplus), input: [H], decay: true, recurrent: [-1])
					let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 384, distribution: .Gauss, activation: .Binary,
					                               adapters: (.Linear, .Softplus), input: [G], decay: true, recurrent: [-1])
					let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width: 256, distribution: .Gauss, activation: .Binary,
					                               adapters: (.Linear, .Softplus), input: [F], decay: true, recurrent: [-1])
					try context.save()
					context.reset()
				}
			}
			try Array<Void>(repeating: (), count: 4).forEach {
				try autoreleasepool {
					guard let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").first else { assertionFailure(); return }
					guard let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").first else { assertionFailure(); return }
					let batch: Int = 1024
					let limit: Int = string.count
					try stride(from: 1, to: limit, by: batch).forEach {
						let beg = string.index(string.startIndex, offsetBy: $0)
						let end = string.index(beg, offsetBy: batch)
						try autoreleasepool {
							(beg..<end).forEach {
								O.collect_refresh()
								I.correct_refresh()
								O.target = string[$0-0].onehot
								I.source = string[$0-1].onehot
								O.collect()
								I.correct()
								let ratio: Double = Double($0) / Double(limit)
								DispatchQueue.main.async {
									progress.doubleValue = ratio
								}

							}
							try context.save()
						}
						let string: String = String(describing: $0)
						DispatchQueue.main.async {
							label.stringValue = string
						}
					}
					"no it was n't black monday but while the new york stock exchange did n't fall apart friday".utf8CString.forEach {
						O.collect_refresh()
						I.source = $0.onehot
						O.collect()
					}
					let predict: Array<CChar> = Array<Void>(repeating: (), count: 4096).map {
						let last: Array<Float> = O.source
						O.collect_refresh()
						I.source = last
						O.collect()
						return last.asOnehot
					}
					try String(cString: predict).write(to: URL(fileURLWithPath: "/tmp/dump.txt"), atomically: true, encoding: .ascii)
					context.reset()
				}
			}
		} catch {
			assertionFailure(String(describing: error))
		}
	}
}
