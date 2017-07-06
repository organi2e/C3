//
//  Capture.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//
/*
import Accelerate
import CoreData

public typealias LaObjet = la_object_t
extension LaObjet {
	public var rows: Int {
		return Int(la_matrix_rows(self))
	}
	public var cols: Int {
		return Int(la_matrix_cols(self))
	}
	public var array: Array<Float> {
		let result: Array<Float> = Array<Float>(repeating: 0, count: Int(la_matrix_rows(self)*la_matrix_cols(self)))
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: result), la_matrix_cols(self), self)
		assert( status == la_status_t(LA_SUCCESS) )
		return result
	}
}
extension LaObjet {
	public func write(to: URL) throws {
		let data: Data = Data(count: rows * cols * MemoryLayout<Float>.size)
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer(mutating: data.withUnsafeBytes{$0}), la_matrix_cols(self), self)
		assert(status==la_status_t(LA_SUCCESS))
		try data.write(to: to)
	}
}
	public func make(value: Float) -> LaObjet {
		return la_splat_from_float(value, attr)
	}
	public func make(value: Float, rows: Int, cols: Int) -> LaObjet {
		return la_matrix_from_splat(la_splat_from_float(value, attr), la_count_t(rows), la_count_t(cols))
	}
	public func make(array: [Float], rows: Int, cols: Int) -> LaObjet {
		assert(array.count==rows*cols)
		return la_matrix_from_float_buffer(array, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
	}
public func rsqrt(_ x: LaObjet) -> LaObjet {
	let rows: la_count_t = la_matrix_rows(x)
	let cols: la_count_t = la_matrix_cols(x)
	return Data(capacity: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes { (ref: UnsafePointer<Float>) -> LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, x)
		assert(status==la_status_t(LA_SUCCESS))
		vvrsqrtf(UnsafeMutablePointer<Float>(mutating: ref), ref, [Int32(rows*cols)])
		return la_matrix_from_float_buffer(ref, rows, cols, cols, hint, attr)
	}
}
public func sqrt(_ x: LaObjet) -> LaObjet {
	let rows: la_count_t = la_matrix_rows(x)
	let cols: la_count_t = la_matrix_cols(x)
	return Data(count: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes { (ref: UnsafePointer<Float>) -> LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, x)
		assert(status==la_status_t(LA_SUCCESS))
		vvsqrtf(UnsafeMutablePointer<Float>(mutating: ref), ref, [Int32(rows*cols)])
		return la_matrix_from_float_buffer(ref, rows, cols, cols, hint, attr)
	}
}
public func exp(_ x: LaObjet) -> LaObjet {
	let rows: la_count_t = la_matrix_rows(x)
	let cols: la_count_t = la_matrix_cols(x)
	return Data(capacity: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes { (ref: UnsafePointer<Float>) -> LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, x)
		assert(status==la_status_t(LA_SUCCESS))
		vvexpf(UnsafeMutablePointer<Float>(mutating: ref), ref, [Int32(rows*cols)])
		return la_matrix_from_float_buffer(ref, rows, cols, cols, hint, attr)
	}
}
public func erf(_ x: LaObjet) -> LaObjet {
	let rows: la_count_t = la_matrix_rows(x)
	let cols: la_count_t = la_matrix_cols(x)
	return Data(capacity: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes { (ref: UnsafePointer<Float>) -> LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, x)
		assert(status==la_status_t(LA_SUCCESS))
		return la_matrix_from_float_buffer(UnsafeBufferPointer<Float>(start: ref, count: Int(rows*cols)).map(erf), rows, cols, cols, hint, attr)
	}
}
public func +(lhs: Float, rhs: LaObjet) -> LaObjet {
	return la_sum(la_splat_from_float(lhs, attr), rhs)
}
public func +(lhs: LaObjet, rhs: Float) -> LaObjet {
	return la_sum(lhs, la_splat_from_float(rhs, attr))
}
public func +(lhs: LaObjet, rhs: LaObjet) -> LaObjet {
	assert( lhs.rows == rhs.rows && lhs.cols == rhs.cols )
	return la_sum(lhs, rhs)
}
public func -(lhs: Float, rhs: LaObjet) -> LaObjet {
	return la_difference(la_splat_from_float(lhs, attr), rhs)
}
public func -(lhs: LaObjet, rhs: Float) -> LaObjet {
	return la_difference(lhs, la_splat_from_float(rhs, attr))
}
public func -(lhs: LaObjet, rhs: LaObjet) -> LaObjet {
	assert( lhs.rows == rhs.rows && lhs.cols == rhs.cols )
	return la_difference(lhs, rhs)
}
public func *(lhs: Float, rhs: LaObjet) -> LaObjet {
	return la_scale_with_float(rhs, lhs)
}
public func *(lhs: LaObjet, rhs: Float) -> LaObjet {
	return la_scale_with_float(lhs, rhs)
}
public func *(lhs: LaObjet, rhs: LaObjet) -> LaObjet {
	assert( lhs.rows == rhs.rows && lhs.cols == rhs.cols )
	return la_elementwise_product(lhs, rhs)
}
public func /(lhs: Float, rhs: LaObjet) -> LaObjet {
	let rows: la_count_t = la_matrix_rows(rhs)
	let cols: la_count_t = la_matrix_cols(rhs)
	return Data(capacity: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes{(ref: UnsafePointer<Float>)->LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, rhs)
		assert(status==la_status_t(LA_SUCCESS))
		vvrecf(UnsafeMutablePointer<Float>(mutating: ref), ref, [Int32(rows*cols)])
		return la_scale_with_float(la_matrix_from_float_buffer(ref, rows, cols, cols, hint, attr), lhs)
	}
}
public func /(lhs: LaObjet, rhs: Float) -> LaObjet {
	return la_scale_with_float(lhs, 1/rhs)
}
public func /(lhs: LaObjet, rhs: LaObjet) -> LaObjet {
	let rows: la_count_t = min(la_matrix_rows(lhs), la_matrix_rows(rhs))
	let cols: la_count_t = min(la_matrix_cols(lhs), la_matrix_cols(rhs))
	assert( lhs.rows == rhs.rows && lhs.cols == rhs.cols )
	return Data(capacity: Int(rows*cols)*MemoryLayout<Float>.size).withUnsafeBytes{(ref: UnsafePointer<Float>)->LaObjet in
		let status: la_status_t = la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: ref), cols, rhs)
		assert(status==la_status_t(LA_SUCCESS))
		vvrecf(UnsafeMutablePointer<Float>(mutating: ref), ref, [Int32(rows*cols)])
		return la_elementwise_product(lhs, la_matrix_from_float_buffer(ref, rows, cols, cols, hint, attr))
	}
}
public func inner_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_inner_product(lhs, rhs)
}
public func outer_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_outer_product(lhs, rhs)
}
public func matrix_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_matrix_product(lhs, rhs)
}
extension Context {
	/*
	public func capture(output: Cell, input: Cell) -> (LaObjet, LaObjet) {
		guard let edge: Edge = output.input.filter({ $0.input.objectID == input.objectID }).first else { return (la_splat_from_float(0, attr), la_splat_from_float(0, attr)) }
		let rows: Int = output.width
		let cols: Int = input.width
		let bytes: Int = rows * cols * MemoryLayout<Float>.size
		let μ: Buffer = make(length: bytes, options: .storageModeShared)
		let σ: Buffer = make(length: bytes, options: .storageModeShared)
		let commandBuffer: CommandBuffer = make()
		edge.access(commandBuffer: commandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: $0.μ, sourceOffset: 0, to: μ, destinationOffset: 0, size: min($0.μ.length, μ.length))
			encoder.copy(from: $0.σ, sourceOffset: 0, to: σ, destinationOffset: 0, size: min($0.σ.length, σ.length))
			encoder.endEncoding()
		}
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
		defer {
			μ.setPurgeableState(.empty)
			σ.setPurgeableState(.empty)
		}
		return (
			la_matrix_from_float_buffer(μ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr),
			la_matrix_from_float_buffer(σ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
		)
	}
	public func capture(cell: Cell) -> (LaObjet, LaObjet) {
		let width: Int = cell.bias.cell.width
		let bytes: Int = width * MemoryLayout<Float>.size
		let μ: Buffer = make(length: bytes, options: .storageModeShared)
		let σ: Buffer = make(length: bytes, options: .storageModeShared)
		let commandBuffer: CommandBuffer = make()
		cell.bias.access(commandBuffer: commandBuffer) {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: $0.μ, sourceOffset: 0, to: μ, destinationOffset: 0, size: min($0.μ.length, μ.length))
			encoder.copy(from: $0.σ, sourceOffset: 0, to: σ, destinationOffset: 0, size: min($0.σ.length, σ.length))
			encoder.endEncoding()
		}
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
		defer {
			μ.setPurgeableState(.empty)
			σ.setPurgeableState(.empty)
		}
		return (
			la_matrix_from_float_buffer(μ.ref, la_count_t(width), 1, 1, hint, attr),
			la_matrix_from_float_buffer(σ.ref, la_count_t(width), 1, 1, hint, attr)
		)
	}
	*/
}
private extension Buffer {
	var buf: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())),
		                                         count: length / MemoryLayout<Float>.size)
	}
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
}
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
*/
