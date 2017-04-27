//
//  MNIST.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/10.
//
//

import Accelerate
import CoreData
import Foundation

extension Educator {
	private static let name: String = "MNIST"
	private static let label: String = "label"
	private static let image: String = "image"
	private static let plist: String = "plist"
	public enum MNIST: String {
		case train = "train"
		case t10k = "t10k"
	}
	public func count(mnist: MNIST, label: IntegerLiteralType? = nil, offset: Int = 0, limit: Int = 0) throws -> Int {
		let handle: String = {
			guard let handle: IntegerLiteralType = $0 else { return "" }
			return String(describing: handle)
		} ( label )
		return try count(domain: type(of: self).name, family: mnist.rawValue, option: [:], handle: handle, offset: offset, limit: limit)
	}
	public func fetch(mnist: MNIST, label: IntegerLiteralType? = nil, offset: Int = 0, limit: Int = 0) throws -> Array<Image> {
		let handle: String = {
			guard let handle: IntegerLiteralType = $0 else { return "" }
			return String(describing: handle)
		} ( label )
		return try fetch(domain: type(of: self).name, family: mnist.rawValue, option: [:], handle: handle, offset: offset, limit: limit)
	}
	public func build(mnist: MNIST) throws {
		let name: String = String(describing: type(of: self).name)
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: type(of: self).plist) else {
			throw ErrorCases.NoResourceFound(name: name, extension: type(of: self).plist)
		}
		guard let dictionary: Dictionary<String, Dictionary<String, String>> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, Dictionary<String, String>> else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let labelString: String = dictionary[mnist.rawValue]?[type(of: self).label] else {
			throw ErrorCases.NoRecourdFound(name: "\(mnist.rawValue).\(type(of: self).label)")
		}
		guard let labelURL: URL = URL(string: labelString) else {
			throw ErrorCases.InvalidFormat(of: labelString, for: URL.self)
		}
		guard let imageString: String = dictionary[mnist.rawValue]?[type(of: self).image] else {
			throw ErrorCases.NoRecourdFound(name: "\(mnist.rawValue).\(type(of: self).image)")
		}
		guard let imageURL: URL = URL(string: imageString) else {
			throw ErrorCases.InvalidFormat(of: imageString, for: URL.self)
		}
		let (labelhead, labelbody): (Data, Data) = try gunzip(data: Data(contentsOf: labelURL)).split(cursor: 2 * MemoryLayout<UInt32>.size)
		let labelheader: Array<Int> = labelhead.toArray().map{Int(UInt32(bigEndian: $0))}
		let labels: Array<UInt8> = labelbody.toArray()
		let (imagehead, imagebody): (Data, Data) = try gunzip(data: Data(contentsOf: imageURL)).split(cursor: 4 * MemoryLayout<UInt32>.size)
		let imageheader: Array<Int> = imagehead.toArray().map{Int(UInt32(bigEndian: $0))}
		guard imageheader.count == 4 else { throw ErrorCases.UnknownError(message: "") }
		guard imageheader[1] == labelheader[1] else { throw ErrorCases.InvalidFormat(of: "image.length", for: "label.length") }
		let length: Int = min(imageheader[1], labelheader[1])
		let rows: Int = imageheader[2]
		let cols: Int = imageheader[3]
		guard length * rows * cols == imagebody.count else { throw ErrorCases.InvalidFormat(of: length * rows * cols, for: imagebody.count) }
		let pixels: Array<Data> = imagebody.chunk(width: Int(rows*cols))
		
		let context: NSManagedObjectContext = NSManagedObjectContext(concurrencyType: .privateQueueConcurrencyType)
		do {
			context.parent = self
		}
		try context.fetch(make(domain: type(of: self).name, family: mnist.rawValue, option: Dictionary<String, Any>(), handle: "", offset: 0, limit: 0)).forEach(context.delete)
		
		let entityName: String = String(describing: Image.self)
		try zip(labels, pixels).forEach {
			guard let image: Image = NSEntityDescription.insertNewObject(forEntityName: entityName, into: context) as? Image else {
				throw ErrorCases.NoEntityFound(name: entityName)
			}
			image.domain = type(of: self).name
			image.family = mnist.rawValue
			image.option = Dictionary<String, Any>()
			image.handle = String(describing: $0.0)
			image.height = UInt16(rows)
			image.width = UInt16(cols)
			image.rowBytes = UInt32(cols)
			image.format = kCIFormatR8
			image.data = $0.1
		}
		try context.save()
	}
}
/*



public class MNISTImage: NSManagedObject {

}
extension MNISTImage {
	@NSManaged var group: String
	@NSManaged var label: UInt16
	@NSManaged var pixel: Data
	@NSManaged var image: vImage_Buffer
}
extension MNISTImage {
	
}
public class MNIST: NSManagedObjectContext {
	public init(storage: URL?) throws {
		let type: String = storage == nil ? NSInMemoryStoreType : storage?.pathExtension == "db" || storage?.pathExtension == "sqlite" ? NSSQLiteStoreType : NSBinaryStoreType
		let name: String = String(describing: type(of: self))
		guard let url: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "momd") else {
			throw ErrorCases.NoResourceFound(name: name, extension: "momd")
		}
		guard let model: NSManagedObjectModel = NSManagedObjectModel(contentsOf: url) else {
			throw ErrorCases.NoModelFound(name: name)
		}
		super.init(concurrencyType: .privateQueueConcurrencyType)
		persistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		try persistentStoreCoordinator?.addPersistentStore(ofType: type, configurationName: nil, at: storage, options: nil)
	}
	public required init?(coder aDecoder: NSCoder) {
		super.init(coder: aDecoder)
	}
}
extension MNIST {
	private static let labelKey: String = "label"
	private static let imageKey: String = "image"
	public enum Group: String {
		case train = "train"
		case t10k = "t10k"
	}
	public func rebuild(group: Group) throws {
		let file: FileManager = FileManager.default
		let name: String = String(describing: type(of: self))
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "plist") else {
			throw ErrorCases.NoResourceFound(name: name, extension: "plist")
		}
		guard let dictionary: Dictionary<String, Dictionary<String, String>> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, Dictionary<String, String>> else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let labelString: String = dictionary[group.rawValue]?[type(of: self).labelKey] else {
			throw ErrorCases.NoRecourdFound(name: "\(group.rawValue).\(type(of: self).labelKey)")
		}
		guard let labelURL: URL = URL(string: labelString) else {
			throw ErrorCases.InvalidFormat(of: labelString, for: URL.self)
		}
		guard let imageString: String = dictionary[group.rawValue]?[type(of: self).imageKey] else {
			throw ErrorCases.NoRecourdFound(name: "\(group.rawValue).\(type(of: self).imageKey)")
		}
		guard let imageURL: URL = URL(string: imageString) else {
			throw ErrorCases.InvalidFormat(of: imageString, for: URL.self)
		}	
		let imageCache: FileHandle = try FileHandle(forReadingFrom: file.download(url: imageURL))
		defer {
			imageCache.closeFile()
		}
		let labelCache: FileHandle = try FileHandle(forReadingFrom: file.download(url: labelURL))
		defer {
			labelCache.closeFile()
		}
		let (labelhead, labelbody): (Data, Data) = try labelCache.gunzip().split(cursor: 2 * MemoryLayout<UInt32>.size)
		let labelheader: Array<Int> = labelhead.toArray().map{Int(UInt32(bigEndian: $0))}
		let labels: Array<UInt8> = labelbody.toArray()
		let (imagehead, imagebody): (Data, Data) = try imageCache.gunzip().split(cursor: 4 * MemoryLayout<UInt32>.size)
		let imageheader: Array<Int> = imagehead.toArray().map{Int(UInt32(bigEndian: $0))}
		guard imageheader.count == 4 else { throw ErrorCases.UnknownError(message: "") }
		guard imageheader[1] == labelheader[1] else { throw ErrorCases.InvalidFormat(of: "image.length", for: "label.length") }
		let length: Int = min(imageheader[1], labelheader[1])
		let rows: Int = imageheader[2]
		let cols: Int = imageheader[3]
		guard length * rows * cols == imagebody.count else { throw ErrorCases.InvalidFormat(of: length * rows * cols, for: imagebody.count) }
		let pixels: Array<Data> = imagebody.chunk(width: rows*cols)
		let entityName: String = String(describing: MNISTImage.self)
		let request: NSFetchRequest<MNISTImage> = NSFetchRequest<MNISTImage>(entityName: entityName)
		request.predicate = NSPredicate(format: "group = %@", group.rawValue)
		try fetch(request).forEach(delete)
		try zip(labels, pixels).forEach {
			guard let image: MNISTImage = NSEntityDescription.insertNewObject(forEntityName: entityName, into: self) as? MNISTImage else { throw ErrorCases.NoEntityFound(name: entityName) }
			image.group = group.rawValue
			image.label = UInt16($0.0)
			image.pixel = $0.1
		}
		
		/*
		let pixel: Array<UInt8> = imagebody.toArray()
		guard length * rows * cols == pixel.count else { throw ErrorCases.InvalidFormat(of: length * rows * cols, for: pixel.count) }
		let pixels: Array<Array<UInt8>> = pixel.chunk(width: rows * cols)
		let entityName: String = String(describing: MNISTImage.self)
		let request: NSFetchRequest<MNISTImage> = NSFetchRequest<MNISTImage>(entityName: entityName)
		request.predicate = NSPredicate(format: "group = %@", group.rawValue)
		try fetch(request).forEach(delete)
		try zip(labels, pixels).forEach {
			guard let image: MNISTImage = NSEntityDescription.insertNewObject(forEntityName: entityName, into: self) as? MNISTImage else { throw ErrorCases.NoEntityFound(name: entityName) }
			image.group = group.rawValue
			image.label = UInt16($0.0)
			image.pixel = Data(bytes: $0.1)
		}
		*/
	}
	public func fetch(group: Group, label: IntegerLiteralType? = nil, offset: Int = 0, limit: Int = 0) throws -> Array<MNISTImage> {
		let entityName: String = String(describing: MNISTImage.self)
		let request: NSFetchRequest<MNISTImage> = NSFetchRequest<MNISTImage>(entityName: entityName)
		let pair: Array<(String, Any)> = [("group = %@", group.rawValue)] + {
			guard let label: IntegerLiteralType = $0 else { return [] }
			return [("label = %@", label)]
		} (label)
		request.predicate = NSPredicate(format: pair.map{$0.0}.joined(separator: " and "), argumentArray: pair.map{$0.1})
		request.fetchOffset = offset
		request.fetchLimit = limit
		return try fetch(request)
	}
	public func count(group: Group) throws -> Int {
		let entityName: String = String(describing: MNISTImage.self)
		let request: NSFetchRequest<MNISTImage> = NSFetchRequest<MNISTImage>(entityName: entityName)
		request.predicate = NSPredicate(format: "group = %@", group.rawValue)
		return try count(for: request)
	}
}
/*
internal class Image: NSManagedObject {
	
}
extension Image {
	@NSManaged var group: String
	@NSManaged var label: UInt16
	@NSManaged var pixel: Data
}
extension Image: Supervised {
	public var answer: Array<Float> {
		let array: Array<Float> = Array<Float>(repeating: 0, count: 10)
		if 0 <= label && label < 10 {
			UnsafeMutablePointer<Float>(mutating: array).advanced(by: Int(label)).pointee = 1
		}
		return array
	}
	public var source: Array<Float> {
		let array: Array<Float> = Array<Float>(repeating: 0, count: pixel.count)
		pixel.withUnsafeBytes {
			vDSP_vfltu8($0, 1, UnsafeMutablePointer<Float>(mutating: array), 1, vDSP_Length(pixel.count))
		}
		return array
	}
	public var ciimage: CIImage {
		return CIImage(bitmapData: pixel, bytesPerRow: 28, size: CGSize(width: 28, height: 28), format: kCIFormatA8, colorSpace: CGColorSpaceCreateDeviceGray())
	}
}
public class MNIST {
	private static let labelKey: String = "label"
	private static let imageKey: String = "image"
	public enum Group: String {
		case train = "train"
		case t10k = "t10k"
	}
	let context: NSManagedObjectContext
	public init(storage: URL?) throws {
		let type: String = storage == nil ? NSInMemoryStoreType : storage?.pathExtension == "db" || storage?.pathExtension == "sqlite" ? NSSQLiteStoreType : NSBinaryStoreType
		let name: String = String(describing: type(of: self))
		guard let url: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "momd") else {
			throw ErrorCases.NoResourceFound(name: name, extension: "momd")
		}
		guard let model: NSManagedObjectModel = NSManagedObjectModel(contentsOf: url) else {
			throw ErrorCases.NoModelFound(name: name)
		}
		context = NSManagedObjectContext(concurrencyType: .privateQueueConcurrencyType)
		context.persistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		try context.persistentStoreCoordinator?.addPersistentStore(ofType: type, configurationName: nil, at: storage, options: nil)
	}
	public func rebuild(group: Group) throws {
		let file: FileManager = FileManager.default
		let name: String = String(describing: type(of: self))
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "plist") else {
			throw ErrorCases.NoResourceFound(name: name, extension: "plist")
		}
		guard let dictionary: Dictionary<String, Dictionary<String, String>> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, Dictionary<String, String>> else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let labelString: String = dictionary[group.rawValue]?[type(of: self).labelKey] else {
			throw ErrorCases.NoRecourdFound(name: "\(group.rawValue).\(type(of: self).labelKey)")
		}
		guard let labelURL: URL = URL(string: labelString) else {
			throw ErrorCases.InvalidFormat(of: labelString, for: URL.self)
		}
		guard let imageString: String = dictionary[group.rawValue]?[type(of: self).imageKey] else {
			throw ErrorCases.NoRecourdFound(name: "\(group.rawValue).\(type(of: self).imageKey)")
		}
		guard let imageURL: URL = URL(string: imageString) else {
			throw ErrorCases.InvalidFormat(of: imageString, for: URL.self)
		}
		let imageCache: FileHandle = try FileHandle(forReadingFrom: file.download(url: imageURL))
		defer {
			imageCache.closeFile()
		}
		let labelCache: FileHandle = try FileHandle(forReadingFrom: file.download(url: labelURL))
		defer {
			labelCache.closeFile()
		}
		let (labelhead, labelbody): (Data, Data) = try labelCache.gunzip().split(cursor: 2 * MemoryLayout<UInt32>.size)
		let labelheader: Array<Int> = labelhead.toArray().map{Int(UInt32(bigEndian: $0))}
		let labels: Array<UInt8> = labelbody.toArray()
		let (imagehead, imagebody): (Data, Data) = try imageCache.gunzip().split(cursor: 4 * MemoryLayout<UInt32>.size)
		let imageheader: Array<Int> = imagehead.toArray().map{Int(UInt32(bigEndian: $0))}
		guard imageheader.count == 4 else { throw ErrorCases.UnknownError(message: "") }
		guard imageheader[1] == labelheader[1] else { throw ErrorCases.InvalidFormat(of: "image.length", for: "label.length") }
		let length: Int = min(imageheader[1], labelheader[1])
		let rows: Int = imageheader[2]
		let cols: Int = imageheader[3]
		let pixel: Array<UInt8> = imagebody.toArray()
		guard length * rows * cols == pixel.count else { throw ErrorCases.InvalidFormat(of: length * rows * cols, for: pixel.count) }
		let pixels: Array<Array<UInt8>> = pixel.chunk(width: rows * cols)
		let entityName: String = String(describing: Image.self)
		let request: NSFetchRequest<Image> = NSFetchRequest<Image>(entityName: entityName)
		request.predicate = NSPredicate(format: "group = %@", group.rawValue)
		try context.fetch(request).forEach(context.delete)
		try zip(labels, pixels).forEach {
			guard let image: Image = NSEntityDescription.insertNewObject(forEntityName: entityName, into: context) as? Image else { throw ErrorCases.NoEntityFound(name: entityName) }
			image.group = group.rawValue
			image.label = UInt16($0.0)
			image.pixel = Data(bytes: $0.1)
		}
	}
	public func fetch(group: Group, label: IntegerLiteralType? = nil, offset: Int = 0, limit: Int = 0) throws -> Array<Supervised> {
		let entityName: String = String(describing: Image.self)
		let request: NSFetchRequest<Image> = NSFetchRequest<Image>(entityName: entityName)
		let pair: Array<(String, Any)> = [("group = %@", group.rawValue)] + {
			guard let label: IntegerLiteralType = $0 else { return [] }
			return [("label = %@", label)]
		} (label)
		request.predicate = NSPredicate(format: pair.map{$0.0}.joined(separator: " and "), argumentArray: pair.map{$0.1})
		request.fetchOffset = offset
		request.fetchLimit = limit
		return try context.fetch(request)
	}
	public func save() throws {
		try context.save()
	}
	public func count(group: Group) throws -> Int {
		let entityName: String = String(describing: Image.self)
		let request: NSFetchRequest<Image> = NSFetchRequest<Image>(entityName: entityName)
		request.predicate = NSPredicate(format: "group = %@", group.rawValue)
		return try context.count(for: request)
	}
}
*/
*/
private extension Data {
	func split(cursor: Int) -> (Data, Data){
		return(subdata(in: startIndex..<startIndex.advanced(by: cursor)), subdata(in: startIndex.advanced(by: cursor)..<endIndex))
	}
	func chunk(width: Int) -> Array<Data> {
		return stride(from: 0, to: count, by: width).map {
			subdata(in: index(startIndex, offsetBy: $0)..<index(startIndex, offsetBy: $0 + width))
		}
	}
	func toArray<T>() -> [T] {
		return withUnsafeBytes {
			Array<T>(UnsafeBufferPointer<T>(start: $0, count: count / MemoryLayout<T>.size))
		}
	}
}
private extension Array {
	func chunk(width: Int) -> [[Element]] {
		return stride(from: 0, to: count, by: width).map{Array(self[startIndex.advanced(by: $0)..<startIndex.advanced(by: $0 + width)])}
	}
}
