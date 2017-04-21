//
//  Educator.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/12.
//
//
import Accelerate
import CoreImage
import CoreData
import Compression
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
public class Group: NSManagedObject {

}
extension Group {
	@NSManaged internal var domain: String
	@NSManaged internal var family: String
	@NSManaged internal var option: Dictionary<String, Any>
	@NSManaged internal var handle: String
}
public extension Group {
	public func onehot<T>(count: Int, value: T) throws -> Array<T> {
		guard let index: Int = Int(handle), 0 <= index && index < count else { throw ErrorCases.InvalidFormat(of: handle, for: count) }
		return Data(count: count * MemoryLayout<T>.size).withUnsafeBytes { (src: UnsafePointer<T>) -> Array<T> in
			UnsafeMutablePointer<T>(mutating: src).advanced(by: index).pointee = value
			return Array<T>(UnsafeBufferPointer<T>(start: src, count: count))
		}
	}
}
public class Corpus: Group {
	
}
internal extension Corpus {
	@NSManaged internal var title: String
	@NSManaged internal var body: String
}
public extension Corpus {
	
}
public class Image: Group {
	public var source: Array<Float> {
		switch format {
		case kCIFormatAf, kCIFormatRf, kCIFormatRGBAf:
			return data.withUnsafeBytes {
				Array<Float>(UnsafeBufferPointer<Float>(start: $0, count: data.count / MemoryLayout<Float>.size))
			}
		case kCIFormatA8, kCIFormatR8, kCIFormatBGRA8, kCIFormatARGB8:
			let result: Array<Float> = Array<Float>(repeating: 0, count: data.count / MemoryLayout<UInt8>.size)
			data.withUnsafeBytes {
				vDSP_vfltu8($0, 1, UnsafeMutablePointer<Float>(mutating: result), 1, vDSP_Length(data.count))
			}
			cblas_sscal(Int32(result.count), 1/256.0, UnsafeMutablePointer<Float>(mutating: result), 1)
			return result
		case kCIFormatA16, kCIFormatR16:
			let result: Array<Float> = Array<Float>(repeating: 0, count: data.count / MemoryLayout<UInt16>.size)
			data.withUnsafeBytes {
				vDSP_vfltu16($0, 1, UnsafeMutablePointer<Float>(mutating: result), 1, vDSP_Length(data.count))
			}
			cblas_sscal(Int32(result.count), 1/65536.0, UnsafeMutablePointer<Float>(mutating: result), 1)
			return result
		default:
			assertionFailure("CIImage format: \(format) has been not implemented")
			return Array<Float>()
		}
	}
}
extension Image {
	@NSManaged internal var width: UInt16
	@NSManaged internal var height: UInt16
	@NSManaged internal var rowBytes: UInt32
	@NSManaged internal var format: CIFormat
	@NSManaged internal var data: Data
}
extension Image {
	public var ciimage: CIImage {
		let colorSpace: CGColorSpace = [kCIFormatA8, kCIFormatAf, kCIFormatR8, kCIFormatRf].contains(format) ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
		return CIImage(bitmapData: data, bytesPerRow: Int(rowBytes), size: CGSize(width: Int(width), height: Int(height)), format: format, colorSpace: colorSpace)
	}
	public func vImage(handle: (vImage_Buffer) -> Void) {
		data.withUnsafeMutableBytes {
			handle(vImage_Buffer(data: $0, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: Int(rowBytes)))
		}
	}
}
public class Educator: NSManagedObjectContext {
	public init(storage: URL?) throws {
		super.init(concurrencyType: .privateQueueConcurrencyType)
		let type: String = storage == nil ? NSInMemoryStoreType : ["db", "sqlite"].filter{$0==storage?.pathExtension}.isEmpty ? NSBinaryStoreType : NSSQLiteStoreType
		let name: String = String(describing: type(of: self))
		guard let url: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "momd") else {
			throw ErrorCases.NoResourceFound(name: name, extension: "momd")
		}
		guard let model: NSManagedObjectModel = NSManagedObjectModel(contentsOf: url) else {
			throw ErrorCases.NoModelFound(name: name)
		}
		persistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		try persistentStoreCoordinator?.addPersistentStore(ofType: type, configurationName: nil, at: storage, options: nil)
	}
	public required init?(coder aDecoder: NSCoder) {
		assertionFailure("init(coder:) has been not implemented")
		return nil
	}
}
extension Educator {
	internal func make<T: Group>(domain: String, family: String, option: Dictionary<String, Any>, handle: String, offset: Int, limit: Int) -> NSFetchRequest<T> {
		let request: NSFetchRequest<T> = NSFetchRequest<T>(entityName: String(describing: Group.self))
		let formats: Array<(String, Any)> = (domain.isEmpty ?[]:[("domain = %@", domain)])+(family.isEmpty ?[]:[("family = %@", family)])+(option.isEmpty ?[]:[("options = %@", option)])+(handle.isEmpty ?[]:[("handle = %@", handle)])
		request.predicate = NSPredicate(format: formats.map{$0.0}.joined(separator: " and "), argumentArray: formats.map{$0.1})
		request.fetchOffset = offset
		request.fetchLimit = limit
		return request
	}
	internal func count(domain: String, family: String, option: Dictionary<String, Any>, handle: String, offset: Int, limit: Int) throws -> Int {
		return try count(for: make(domain: domain, family: family, option: option, handle: handle, offset: offset, limit: limit))
	}
	internal func fetch<T: Group>(domain: String, family: String, option: Dictionary<String, Any>, handle: String, offset: Int, limit: Int) throws -> Array<T> {
		return try fetch(make(domain: domain, family: family, option: option, handle: handle, offset: offset, limit: limit))
	}
}
internal extension Educator {
	internal func gunzip(data: Data) throws -> Data {
		//reference: http://www.onicos.com/staff/iz/formats/gzip.html
		
		return try data.withUnsafeBytes { (head: UnsafePointer<UInt8>) -> Data in
			
			var seek: UnsafePointer<UInt8> = head
			
			let magic: UInt16 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: magic))
			guard magic == 35615 else { throw ErrorCases.InvalidFormat(of: magic, for: "magic") }
			
			let method: UInt8 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: method))
			guard method == 8 else { throw ErrorCases.InvalidFormat(of: method, for: "method") }
			
			let flags: UInt8 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: flags))
			
			let time: UInt32 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: time))
			
			let extra: UInt8 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: extra))
			
			let os: UInt8 = UnsafePointer(OpaquePointer(seek)).pointee
			seek = seek.advanced(by: MemoryLayout.size(ofValue: os))
			
			guard flags & ( 1 << 1 ) == 0 else { throw ErrorCases.NoImplemented(feature: "multipart") }
			
			let field: Data = {
				guard 0 < ( flags & ( 1 << 2 ) ) else { return Data() }
				let bytes: UInt16 = UnsafePointer(OpaquePointer(seek)).pointee
				seek = seek.advanced(by: MemoryLayout.size(ofValue: bytes))
				let result: Data = Data(bytes: seek, count: Int(bytes))
				seek = seek.advanced(by: result.count)
				return result
			}()

			let original: String = {
				guard 0 < ( flags & ( 1 << 3 )) else { return "" }
				let string: String = String(cString: seek)
				seek = seek.advanced(by: string.characters.count + 1)
				return string
			}()
			
			let comment: String = {
				guard 0 < ( flags & ( 1 << 4 )) else { return "" }
				let string: String = String(cString: seek)
				seek = seek.advanced(by: string.characters.count + 1)
				return string
			}()
			
			let bs: Int = 65536
			return try Data(capacity: bs + MemoryLayout<compression_stream>.size).withUnsafeBytes { (cache: UnsafePointer<UInt8>) -> Data in
				let ref: UnsafeMutablePointer<compression_stream> = UnsafeMutablePointer<compression_stream>(OpaquePointer(cache.advanced(by: bs)))
				guard compression_stream_init(ref, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else {
					throw ErrorCases.UnknownError(message: "ZLIB Initialization")
				}
				defer {
					guard compression_stream_destroy(ref) == COMPRESSION_STATUS_OK else { fatalError() }
				}
				var result: Data = Data()
				ref.pointee.src_ptr = seek
				ref.pointee.src_size = data.count - head.distance(to: seek)
				while true {
					ref.pointee.dst_ptr = UnsafeMutablePointer<UInt8>(mutating: cache)
					ref.pointee.dst_size = bs
					switch compression_stream_process(ref, 0) {
					case COMPRESSION_STATUS_OK:
						result.append(cache, count: bs - ref.pointee.dst_size)
					case COMPRESSION_STATUS_END:
						result.append(cache, count: bs - ref.pointee.dst_size)
						return result
					case COMPRESSION_STATUS_ERROR:
						throw ErrorCases.UnknownError(message: "ZLIB process")
					default:
						fatalError()
					}
				}
			}
		}
	}
}
private extension Data {
	func split(cursor: Int) -> (Data, Data) {
		return (subdata(in: 0..<cursor), subdata(in: cursor..<count))
	}
	func get<T>() -> T {
		assert( MemoryLayout<T>.size <= count )
		return withUnsafeBytes {
			$0.pointee
		}
	}
}
/*
public protocol Image {
	var ciimage: CIImage { get }
}
*/
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
public extension String {
	public func onehot<T>(count: Int, value: T) throws -> Array<T> {
		guard let index: Int = Int(self), 0 <= index && index < count else { throw ErrorCases.InvalidFormat(of: self, for: count) }
		return Data(count: count * MemoryLayout<T>.size).withUnsafeBytes { (src: UnsafePointer<T>) -> Array<T> in
			UnsafeMutablePointer<T>(mutating: src).advanced(by: index).pointee = value
			return Array<T>(UnsafeBufferPointer<T>(start: src, count: count))
		}
	}
}
public extension IntegerLiteralType {
	public func onehot<T>(count: Int, value: T) throws -> Array<T> {
		return Data(count: count * MemoryLayout<T>.size).withUnsafeBytes { (src: UnsafePointer<T>) -> Array<T> in
			UnsafeMutablePointer<T>(mutating: src).advanced(by: self).pointee = value
			return Array<T>(UnsafeBufferPointer<T>(start: src, count: count))
		}
	}
}
