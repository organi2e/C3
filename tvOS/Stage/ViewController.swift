//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 2017/04/04.
//
//

import UIKit
import C3
import Optimizer
import Educator
import Metal

let prefix: String = "GaussAE-"
//let prefix: String = "DegenerateAE-"
let suffix: String = "vae"
let trainer: URL = FileManager.default.temporaryDirectory.appendingPathComponent("trainer.sqlite")
let storage: URL = FileManager.default.temporaryDirectory.appendingPathComponent("MNISTv2.sqlite")

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
				let context: Context = try Context(queue: queue,
				                                   storage: storage,
				                                   optimizer: SMORMS3.factory(L2: 1e-6, α: 1e-3))
				let educator: Educator = try Educator(storage: trainer)
				if try 0 == educator.count(family: .train) {
					try educator.build(family: .train)
					try educator.save()
					print("build")
				}
				if try 0 == context.count(label: "\(prefix)I\(suffix)") {
					print("insert")
					let I: Cell = try context.make(label: "\(prefix)I\(suffix)", width: 10,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor))
					let H: Cell = try context.make(label: "\(prefix)H\(suffix)", width: 32,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [I])
					let G: Cell = try context.make(label: "\(prefix)G\(suffix)", width: 128,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [H])
					let F: Cell = try context.make(label: "\(prefix)F\(suffix)", width: 512,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [G])
					let E: Cell = try context.make(label: "\(prefix)E\(suffix)", width: 28 * 28,
					                               distribution: .Degenerate, activation: .Identity, input: [F])
					let D: Cell = try context.make(label: "\(prefix)D\(suffix)", width: 512,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [E])
					let C: Cell = try context.make(label: "\(prefix)C\(suffix)", width: 128,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [D])
					let B: Cell = try context.make(label: "\(prefix)B\(suffix)", width: 32,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [C])
					let _: Cell = try context.make(label: "\(prefix)A\(suffix)", width: 10,
					                               distribution: .Gauss, activation: .Binary, adapters: (.Regular, .RegFloor), input: [B])
					try context.save()
					context.reset()
				}
				try (0..<4).forEach {
					let label: String = String($0)
					DispatchQueue.main.async {
						lab.text = label
					}
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
						let count: Int = try educator.count(family: .train)
						let batch: Int = 10000
						try stride(from: 0, to: count, by: batch).forEach {
							let ratio: Float = Float($0) / Float(count)
							DispatchQueue.main.async {
								bar.progress = ratio
							}
							try educator.fetch(family: .train, offset: $0, limit: batch).forEach {
								
								I.pliable = false
								H.pliable = false
								G.pliable = false
								F.pliable = false
								E.pliable = false
								D.pliable = true
								C.pliable = true
								B.pliable = true
								A.pliable = true
								
								A.collect_refresh()
								E.correct_refresh()
								A.target = try $0.onehot(count: 10, value: 1)
								E.source = $0.source
								A.collect()
								E.correct()
								
								I.pliable = false
								H.pliable = true
								G.pliable = true
								F.pliable = true
								E.pliable = true
								D.pliable = false
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
						let E: Cell = try context.fetch(label: "\(prefix)E\(suffix)").last else {
							assertionFailure()
							return
					}
					let output: URL = FileManager.default.temporaryDirectory
					try (0..<60).forEach { index in
						E.collect_refresh()
						I.source = (0..<10).map {
							Float( $0 == index % 10 ? 1 : 0 )
						}
						E.collect()
						try Data(buffer: UnsafeBufferPointer<Float>(start: E.source, count: 28 * 28))
							.write(to: output.appendingPathComponent("img\(index).raw"), options: [])
					}
					print(output)
				}
			} catch {
				print(String(describing: error))
			}
		}.start()
		// Do any additional setup after loading the view, typically from a nib.
	}

	override func didReceiveMemoryWarning() {
		super.didReceiveMemoryWarning()
		// Dispose of any resources that can be recreated.
	}
}

