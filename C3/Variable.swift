//
//  Variable.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/04/06.
//
//

import Adapter
import Optimizer
internal struct Variable {
	internal let Δ: Buffer
	internal let θ: Buffer
	private let φ: Buffer
	private let a: Adapter
	private let o: Optimizer
	init(context: Context, data: Data, adapter: Adapter, optimizer: Optimizer) {
		Δ = context.make(length: data.count, options: .storageModePrivate)
		θ = context.make(length: data.count, options: .storageModePrivate)
		φ = context.make(data: data, options: .storageModeShared)
		a = adapter
		o = optimizer
	}
	func flush(commandBuffer: CommandBuffer) {
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: Δ, range: NSRange(location: 0, length: Δ.length), value: 0)
		encoder.endEncoding()
	}
	func refresh(commandBuffer: CommandBuffer) {
		a.generate(commandBuffer: commandBuffer, θ: θ, φ: φ)
	}
	func update(commandBuffer: CommandBuffer) {
		a.gradient(commandBuffer: commandBuffer, Δ: Δ, θ: θ, φ: φ)
		o.optimize(commandBuffer: commandBuffer, θ: φ, Δ: Δ)
	}
	func reset(commandBuffer: CommandBuffer) {
		o.reset(commandBuffer: commandBuffer)
	}
	var data: Data {
		return Data(bytesNoCopy: φ.contents(), count: φ.length, deallocator: .none)
	}
}
