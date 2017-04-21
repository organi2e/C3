//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import UIKit
import C3
import Optimizer
import Educator
import Metal

let prefix: String = "GaussAE-"
//let prefix: String = "DegenerateAE-"
let suffix: String = "v3.1"

class ViewController: UIViewController {
	
	@IBOutlet weak var bar: UIProgressView?
	@IBOutlet weak var lab: UILabel?
	
	override func viewDidLoad() {
		super.viewDidLoad()
		guard
			let bar: UIProgressView = bar,
			let lab: UILabel = lab,
			let device: MTLDevice = MTLCreateSystemDefaultDevice() else {
				assertionFailure()
				return
		}
		let queue: MTLCommandQueue = device.makeCommandQueue()
		Thread {
			do {
				let trainer: URL = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("trainer.sqlite")
				let educator: Educator = try Educator(storage: trainer)
				if try 0 == educator.count(family: .train) {
					print("build")
					try educator.build(family: .train)
					try educator.save()
				}
				let storage: URL = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("MNISTv2.sqlite")
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L2: 1e-6, α: 1e-3))
				if try 0 == context.count(label: "\(prefix)I\(suffix)") {
					try autoreleasepool {
						print("insert")
						let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 10, distribution: .Gauss, activation: .Identity, adapters: (.Regular, .RegFloor))
						let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 64, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [I])
						let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 64, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [H])
						let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 128, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [G])
						let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 128, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [F])
						let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 256, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [E])
						let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 256, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [D])
						let B: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 512, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [C])
						let A: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 512, distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [B])
						let _: Cell = try context.make(label: "\(prefix)O\(suffix)", width: 28 * 28, distribution: .Gauss, activation: .Identity, adapters: (.Regular, .RegFloor), input: [A])
						try context.save()
					}
					context.reset()
				}
				guard
					let I: Cell = try context.fetch(label: "\(prefix)I\(suffix)").last,
					let O: Cell = try context.fetch(label: "\(prefix)O\(suffix)").last else {
						assertionFailure()
						return
				}
				I.pliable = false
				let count: Int = try educator.count(family: .train)
				let batch: Int = 1000
				try (0..<1000).forEach {
					let label: String = String(describing: $0)
					DispatchQueue.main.async {
						lab.text = label
					}
					try autoreleasepool {
						try stride(from: 0, to: count, by: batch).forEach {
							let ratio: Float = Float($0) / Float(count)
							DispatchQueue.main.async {
								bar.progress = ratio
							}
							try educator.fetch(family: .train, offset: $0, limit: batch).forEach {
								O.collect_refresh()
								I.correct_refresh()
								O.target = $0.source
								I.source = try $0.onehot(count: 10, value: 1)
								O.collect()
								I.correct()
							}
							try context.save()
							educator.reset()
						}
					}
					context.refreshAllObjects()
				}
				do {
					let output: URL = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
					try (0..<60).forEach { index in
						O.collect_refresh()
						I.source = (0..<10).map {
							Float( $0 == index % 10 ? 1 : 0 )
						}
						O.collect()
						try Data(buffer: UnsafeBufferPointer<Float>(start: O.source, count: 28 * 28))
							.write(to: output.appendingPathComponent("img\(index).raw"), options: [])
					}
				}
				print("end")
			} catch {
				fatalError(String(describing: error))
			}
			}.start()
		// Do any additional setup after loading the view, typically from a nib.
	}
	
	override func didReceiveMemoryWarning() {
		super.didReceiveMemoryWarning()
		// Dispose of any resources that can be recreated.
	}
}

