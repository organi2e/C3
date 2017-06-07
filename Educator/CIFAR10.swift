//
//  Cifar-10.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/21.
//
//

import Accelerate
import CoreData

extension Educator {
	private static let domain: String = "CIFAR10"
	private static let ROWSKey: String = "ROWS"
	private static let COLSKey: String = "COLS"
	private static let METAKey: String = "META"
	private static let PATHKey: String = "PATH"
	private static let plist: String = "plist"
	private static let label: String = "label"
	private static let image: String = "image"
	public enum CIFAR10: String {
		case databatch1 = "DATA BATCH 1"
		case databatch2 = "DATA BATCH 2"
		case databatch3 = "DATA BATCH 3"
		case databatch4 = "DATA BATCH 4"
		case databatch5 = "DATA BATCH 5"
		case databatch6 = "DATA BATCH 6"
		case testbatch = "TEST BATCH"
	}
	public func count(cifar10: CIFAR10, handle: String = "", offset: Int = 0, limit: Int = 0) throws -> Int {
		return try count(domain: type(of: self).domain, family: cifar10.rawValue, option: [:], handle: handle, offset: offset, limit: limit)
	}
	public func fetch(cifar10: CIFAR10, handle: String = "", offset: Int = 0, limit: Int = 0) throws -> Array<Image> {
		return try fetch(domain: type(of: self).domain, family: cifar10.rawValue, option: [:], handle: handle, offset: offset, limit: limit)
	}
	public func build(cifar10: CIFAR10) throws {
		let name: String = String(describing: type(of: self).domain)
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: type(of: self).plist) else {
			throw ErrorCases.NoResourceFound(name: name, extension: type(of: self).plist)
		}
		guard let dictionary: Dictionary<String, Any> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, Any> else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let rows: Int = dictionary[type(of: self).ROWSKey] as? Int else {
			throw ErrorCases.NoRecourdFound(name: type(of: self).ROWSKey)
		}
		guard let cols: Int = dictionary[type(of: self).COLSKey] as? Int else {
			throw ErrorCases.NoRecourdFound(name: type(of: self).COLSKey)
		}
		guard let meta: String = dictionary[type(of: self).METAKey] as? String else {
			throw ErrorCases.NoRecourdFound(name: type(of: self).METAKey)
		}
		guard let path: String = dictionary[cifar10.rawValue] as? String else {
			throw ErrorCases.NoRecourdFound(name: cifar10.rawValue)
		}
		guard let binaryPath: String = dictionary[type(of: self).PATHKey] as? String else {
			throw ErrorCases.NoRecourdFound(name: type(of: self).PATHKey)
		}
		guard let binaryURL: URL = URL(string: binaryPath) else {
			throw ErrorCases.InvalidFormat(of: binaryPath, for: URL.self)
		}
		var error: Error?
		var store: URL?
		let semaphore: DispatchSemaphore = DispatchSemaphore(value: 0)
		URLSession.shared.downloadTask(with: URLRequest(url: binaryURL, cachePolicy: .returnCacheDataElseLoad)) {
			error = $2
			store = $0
			semaphore.signal()
		}.resume()
		let context: NSManagedObjectContext = NSManagedObjectContext(concurrencyType: .privateQueueConcurrencyType)
		do {
			context.parent = self
		}
		try context.fetch(make(domain: type(of: self).domain, family: cifar10.rawValue, option: Dictionary<String, Any>(), handle: "", offset: 0, limit: 0)).forEach(context.delete)
		semaphore.wait()
		if let error: Error = error {
			throw error
		}
		guard let file: URL = store else {
			throw ErrorCases.NoFileDownload(from: binaryURL)
		}
		var labels: Dictionary<UInt8, String> = Dictionary<UInt8, String>()
		try FileHandle(forReadingFrom: file).gunzip().untar {
			if $0.isEmpty || $1.isEmpty {
			
			} else if $0 == meta {
				guard let text: String = String(data: $1, encoding: .utf8) else {
					throw ErrorCases.InvalidFormat(of: $1, for: type(of: self).METAKey)
				}
				text.components(separatedBy: .newlines).enumerated().forEach {
					labels.updateValue($0.element, forKey: UInt8($0.offset))
				}
			} else if $0 == path {
				try $1.chunk(width: rows * cols * 3 + 1).forEach {
					let(head, tail): (Data, Data) = $0.split(cursor: 1)
					guard tail.count == rows * cols * 3 else {
						throw ErrorCases.InvalidFormat(of: $0.count, for: type(of: self).image)
					}
					let entityName: String = String(describing: Image.self)
					guard let image: Image = NSEntityDescription.insertNewObject(forEntityName: entityName, into: context) as? Image else {
						throw ErrorCases.NoEntityFound(name: entityName)
					}
					image.domain = type(of: self).domain
					image.family = cifar10.rawValue
					image.option = Dictionary<String, Any>(dictionaryLiteral: (type(of: self).label, head.get() as UInt8))
					
					image.height = UInt16(rows)
					image.width = UInt16(cols)
					image.rowBytes = UInt32(4*cols)
					image.format = kCIFormatBGRA8
					image.data = Data(count: Int(image.height) * Int(image.rowBytes))
					image.data.withUnsafeMutableBytes { (data: UnsafeMutablePointer<UInt8>) in
						let height: vImagePixelCount = vImagePixelCount(rows)
						let width: vImagePixelCount = vImagePixelCount(cols)
						let result: Int = tail.withUnsafeBytes {
							vImageConvert_Planar8ToBGRX8888(
								[vImage_Buffer(data: UnsafeMutablePointer<UInt8>(mutating: $0).advanced(by: 2*rows*cols), height: height, width: width, rowBytes: cols)],
								[vImage_Buffer(data: UnsafeMutablePointer<UInt8>(mutating: $0).advanced(by: 1*rows*cols), height: height, width: width, rowBytes: cols)],
								[vImage_Buffer(data: UnsafeMutablePointer<UInt8>(mutating: $0).advanced(by: 0*rows*cols), height: height, width: width, rowBytes: cols)],
								255,
								[vImage_Buffer(data: data, height: height, width: width, rowBytes: 4 * cols)],
								0)
						}
						assert(result == kvImageNoError)
					}
				}
			}
		}
		try context.insertedObjects.forEach {
			guard
				let image: Image = $0 as? Image,
				let label: UInt8 = image.option[type(of: self).label] as? UInt8,
				let handle: String = labels[label] else {
					throw ErrorCases.NoRecourdFound(name: type(of: self).label)
			}
			image.handle = handle
		}
		try context.save()
	}
}
private extension Data {
	func get<T>() -> T {
		return withUnsafeBytes {
			$0.pointee
		}
	}
	func split(cursor: Int) -> (Data, Data){
		return(subdata(in: startIndex..<startIndex.advanced(by: cursor)), subdata(in: startIndex.advanced(by: cursor)..<endIndex))
	}
	func chunk(width: Int) -> Array<Data> {
		return stride(from: 0, to: count, by: width).map {
			subdata(in: index(startIndex, offsetBy: $0)..<index(startIndex, offsetBy: $0 + width))
		}
	}
}
