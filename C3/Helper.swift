//
//  Helper.swift
//  macOS
//
//  Created by Kota Nakano on 2017/07/05.
//
//

import Accelerate
internal extension Data {
	private var length: Int {
		return count / MemoryLayout<Float>.stride
	}
	mutating func normal(μ: Float, σ: Float) {
		assert( MemoryLayout<Float>.size == MemoryLayout<UInt32>.size )
		withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) in
			arc4random_buf(ref, count)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(length))
			vDSP_vsmsa(ref, 1, [exp2f(-32)], [exp2f(-33)], ref, 1, vDSP_Length(length))
			cblas_sscal(Int32(length/2), 2*Float.pi, ref.advanced(by: length/2), 1)
			vvlogf(ref, ref, [Int32(length/2)])
			cblas_sscal(Int32(length/2), -2, ref, 1)
			vvsqrtf(ref, ref, [Int32(length/2)])
			vDSP_vswap(ref.advanced(by: 1), 2, ref.advanced(by: length/2), 2, vDSP_Length(length/4))
			vDSP_rect(ref, 2, ref, 2, vDSP_Length(length/2))
			vDSP_vsmsa(ref, 1, [σ], [μ], ref, 1, vDSP_Length(length))
		}
	}
	mutating func uniform(α: Float, β: Float) {
		assert( MemoryLayout<Float>.stride == MemoryLayout<UInt32>.stride )
		withUnsafeMutableBytes { (ref: UnsafeMutablePointer<Float>) in
			arc4random_buf(ref, count)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(ref)), 1, ref, 1, vDSP_Length(length))
			vDSP_vsmsa(ref, 1, [(β-α)*exp2f(-32)], [α], ref, 1, vDSP_Length(length))
		}
	}
	mutating func fill(const value: Float) {
		withUnsafeMutableBytes {
			vDSP_vfill([value], $0, 1, vDSP_Length(length))
		}
	}
}
extension Array where Element == Float {
	var objet: LaObjet {
		return la_matrix_from_float_buffer(self, la_count_t(count), 1, 1, la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
	}
}
extension LaObjet {
	var array: Array<Float> {
		let array: Array<Float> = Array<Float>(repeating: 0, count: Int(la_vector_length(self)))
		la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: array), 1, self)
		return array
	}
}
extension Buffer {
	var objet: LaObjet {
		return la_matrix_from_float_buffer_nocopy(UnsafeMutablePointer<Float>(OpaquePointer(contents())), la_count_t(length/MemoryLayout<Float>.stride), 1, 1, hint, nil, attr)
	}
	var array: Array<Float> {
		return Array<Float>(UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.stride))
	}
}
internal typealias LaObjet = la_object_t
internal typealias Device = MTLDevice
internal typealias Buffer = MTLBuffer
internal typealias CommandQueue = MTLCommandQueue
internal typealias CommandBuffer = MTLCommandBuffer
internal typealias ManagedObjectContext = NSManagedObjectContext
internal typealias BlitCommandEncoder = MTLBlitCommandEncoder
internal typealias ComputeCommandEncoder = MTLComputeCommandEncoder
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_DEFAULT_ATTRIBUTES)
