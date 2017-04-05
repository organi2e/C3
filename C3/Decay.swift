//
//  Decay.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//

import CoreData
import Adapter
import Optimizer
internal class Decay: ManagedObject {
	var r: Variable?
}
extension Decay {
	struct Struct {
		let o: Optimizer
		init(context: Context, count: Int) {
			o = context.optimizerFactory(count)
		}
	}
	func setup(commandBuffer: CommandBuffer, count: Int) {
		
	}
}
extension Decay {
	@NSManaged var rate: Data
	@NSManaged var cell: Cell
}
extension Context {
	@nonobjc func make(commandBuffer: CommandBuffer, cell: Cell) throws -> Decay {
		typealias T = Float
		let count: Int = cell.width
		let decay: Decay = try make()
		decay.cell = cell
		decay.rate = Data(count: count * MemoryLayout<T>.size)
		decay.setup(commandBuffer: commandBuffer, count: count)
		return decay
	}
}
