//
//  RingBuffer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

internal struct RingBuffer<T> {
	let buffer: Array<T>
	var offset: Int
	mutating func rotate() {
		let count: Int = buffer.count
		offset = ( offset + 1 ) % count
	}
	var count: Int {
		return buffer.count
	}
	var isEmpty: Bool {
		return buffer.isEmpty
	}
	subscript(index: Int) -> T {
		let count: Int = buffer.count
		return buffer[(((index+offset)%count)+count)%count]
	}
}
