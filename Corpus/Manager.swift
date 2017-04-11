//
//  Manager.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/11.
//
//

import Foundation
import Compression

internal class Manager {
	private enum ErrorCases: Error {
		case IncorrectFormat(value: String)
		case NotImplemented(feature: String)
		case UnknownError(message: String)
	}
	let manager: FileManager
	let cacheDir: URL
	init(directory: String) throws {
		manager = FileManager.default
		cacheDir = try manager.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
		try manager.createDirectory(at: cacheDir, withIntermediateDirectories: true, attributes: nil)
	}
	private func load(cache: String, fetch: String) throws -> FileHandle {
		let cacheURL: URL = cacheDir.appendingPathComponent(cache)
		if !manager.fileExists(atPath: cacheURL.path) {
			guard let fetchURL: URL = URL(string: fetch) else { throw ErrorCases.IncorrectFormat(value: fetch) }
			try Data(contentsOf: fetchURL).write(to: cacheURL)
		}
		return try FileHandle(forReadingFrom: cacheURL)
	}
	internal func gunzip(cache: String, fetch: String) throws -> Data {
		return try gunzip(file: try load(cache: cache, fetch: fetch))
	}
	//reference: http://www.onicos.com/staff/iz/formats/gzip.html
	private func gunzip(file: FileHandle) throws -> Data {
		
		let magic: UInt16 = file.readElement()
		guard magic == 35615 else { throw ErrorCases.IncorrectFormat(value: "magic") }
		
		let method: UInt8 = file.readElement()
		guard method == 8 else { throw ErrorCases.IncorrectFormat(value: "not deflated file") }
		
		let flags: UInt8 = file.readElement()
		guard flags & ( 1 << 1 ) == 0 else { throw ErrorCases.NotImplemented(feature: "multipart") }
		
		let time: UInt32 = file.readElement()
		
		let extra: UInt8 = file.readElement()
		
		let os: UInt8 = file.readElement()
		
		let field: Data = {
			guard 0 < ( flags & ( 1 << 2 ) ) else { return Data() }
			let bytes: UInt16 = file.readElement()
			return file.readData(ofLength: Int(bytes))
		}()
		
		let original: String = {
			guard 0 < ( flags & ( 1 << 3 )) else { return "" }
			return file.readString()
		}()
		
		let comment: String = {
			guard 0 < ( flags & ( 1 << 4 )) else { return "" }
			return file.readString()
		}()
		
		let data: Data = file.readDataToEndOfFile()
		return try data.withUnsafeBytes { (head: UnsafePointer<UInt8>) -> Data in
			
			var pointer: UnsafePointer<UInt8> = head
			
			let bs: Int = compression_decode_scratch_buffer_size(COMPRESSION_ZLIB)
			return try Data(capacity: bs + MemoryLayout<compression_stream>.size).withUnsafeBytes { (cache: UnsafePointer<UInt8>) -> Data in
				let ref: UnsafeMutablePointer<compression_stream> = UnsafeMutablePointer<compression_stream>(OpaquePointer(cache.advanced(by: bs)))
				let buf: UnsafeMutablePointer<UInt8> = UnsafeMutablePointer<UInt8>(mutating: cache)
				guard compression_stream_init(ref, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { throw ErrorCases.UnknownError(message: "compression") }
				defer {
					guard compression_stream_destroy(ref) == COMPRESSION_STATUS_OK else { fatalError("Die") }
				}
				ref.pointee.src_ptr = pointer
				ref.pointee.src_size = data.count - head.distance(to: pointer)
				var result: Data = Data()
				while true {
					ref.pointee.dst_ptr = buf
					ref.pointee.dst_size = bs
					switch compression_stream_process(ref, 0) {
					case COMPRESSION_STATUS_OK:
						result.append(buf, count: bs - ref.pointee.dst_size)
					case COMPRESSION_STATUS_END:
						result.append(buf, count: bs - ref.pointee.dst_size)
						return result
					case COMPRESSION_STATUS_ERROR:
						throw ErrorCases.UnknownError(message: "gunzip parser")
					default:
						fatalError("Die")
					}
				}
			}
		}
	}
}
private extension FileHandle {
	func readString() -> String {
		func recursive(fileHandle: FileHandle) -> Array<CChar> {
			let char: CChar = fileHandle.readElement()
			return Array<CChar>(arrayLiteral: char) + ( char == 0 ? Array<CChar>() : recursive(fileHandle: fileHandle) )
		}
		return String(cString: recursive(fileHandle: self))
	}
	func readElement<T>() -> T {
		return readData(ofLength: MemoryLayout<T>.size).withUnsafeBytes {
			$0.pointee
		}
	}
}
