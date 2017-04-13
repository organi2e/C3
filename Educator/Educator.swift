//
//  Educator.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/12.
//
//

import Compression
import Foundation
internal enum ErrorCases: Error {
	case NoModelFound(name: String)
	case NoPlistFound(name: String)
	case NoEntityFound(name: String)
	case NoRecourdFound(name: String)
	case NoResourceFound(name: String, extension: String)
	case InvalidFormat(of: Any, for: Any)
	case NoImplemented(feature: String)
	case UnknownError(message: String)
}
public protocol Supervised {
	var answer: Array<Float> { get }
	var source: Array<Float> { get }
}
internal extension FileManager {
	func download(url: URL) throws -> URL {
		let cache: URL = temporaryDirectory.appendingPathComponent(UUID().uuidString)
		try Data(contentsOf: url).write(to: cache)
		return cache
	}
}
internal extension FileHandle {
	//reference: http://www.onicos.com/staff/iz/formats/gzip.html
	func gunzip() throws -> Data {
		
		seek(toFileOffset: 0)
		
		let magic: UInt16 = readElement()
		guard magic == 35615 else { throw ErrorCases.InvalidFormat(of: magic, for: "magic") }
		
		let method: UInt8 = readElement()
		guard method == 8 else { throw ErrorCases.InvalidFormat(of: method, for: "method") }
		
		let flags: UInt8 = readElement()
		
		let time: UInt32 = readElement()
		
		let extra: UInt8 = readElement()
		
		let os: UInt8 = readElement()
		
		guard flags & ( 1 << 1 ) == 0 else { throw ErrorCases.NoImplemented(feature: "multipart") }
		
		let field: Data = {
			guard 0 < ( flags & ( 1 << 2 ) ) else { return Data() }
			let bytes: UInt16 = readElement()
			return readData(ofLength: Int(bytes))
		}()
		
		let original: String = {
			guard 0 < ( flags & ( 1 << 3 )) else { return "" }
			return readString()
		}()
		
		let comment: String = {
			guard 0 < ( flags & ( 1 << 4 )) else { return "" }
			return readString()
		}()
		
		let data: Data = readDataToEndOfFile()
		return try data.withUnsafeBytes { (src: UnsafePointer<UInt8>) -> Data in
			let bs: Int = compression_decode_scratch_buffer_size(COMPRESSION_ZLIB)
			return try Data(capacity: bs + MemoryLayout<compression_stream>.size).withUnsafeBytes { (cache: UnsafePointer<UInt8>) -> Data in
				let ref: UnsafeMutablePointer<compression_stream> = UnsafeMutablePointer<compression_stream>(OpaquePointer(cache.advanced(by: bs)))
				let buf: UnsafeMutablePointer<UInt8> = UnsafeMutablePointer<UInt8>(mutating: cache)
				guard compression_stream_init(ref, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { throw ErrorCases.UnknownError(message: "compression") }
				defer {
					guard compression_stream_destroy(ref) == COMPRESSION_STATUS_OK else { fatalError("Die") }
				}
				ref.pointee.src_ptr = src
				ref.pointee.src_size = data.count
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
