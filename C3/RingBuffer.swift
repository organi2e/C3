//
//  RingBuffer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

struct RingBuffer<T> {
	let buffer: Array<T>
	var offset: Int
	mutating func rotate() -> T {
		let count: Int = buffer.count
		offset = ( offset + 1 ) % count
		return buffer[offset%count]
	}
	subscript(index: Int) -> T {
		let count: Int = buffer.count
		return buffer[(((index+offset)%count)+count)%count]
	}
}
