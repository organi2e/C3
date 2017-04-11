//
//  MNIST.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/10.
//
//

import Accelerate
import Foundation

public class MNIST {
	private enum ErrorCases: Error {
		case NoEntityFound(key: String)
		case IncorrectFormat
		case UnknownError
	}
	private struct MNISTImage: Image {
		public let rows: Int
		public let cols: Int
		public let label: Int
		public let pixel: Array<UInt8>
		public var channel: Int {
			return 1
		}
		public var vImage: vImage_Buffer {
			return vImage_Buffer(data: UnsafeMutablePointer(mutating: pixel), height: vImagePixelCount(rows), width: vImagePixelCount(cols), rowBytes: cols)
		}
	}
	public enum Set: String {
		case train = "train"
		case t10k = "t10k"
	}
	private static let directoryKey: String = "Directory"
	private static let imageKey: String = "images"
	private static let labelKey: String = "labels"
	private static let cacheKey: String = "cache"
	private static let fetchKey: String = "fetch"
	public static func load(set: Set) throws -> Array<Image> {
		guard let plist: URL = Bundle(for: self).url(forResource: "Corpus", withExtension: "plist") else { throw ErrorCases.UnknownError }
		guard let dictionary: Dictionary<String, Any> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, Any> else { throw ErrorCases.NoEntityFound(key: "Corpus") }
		guard let mnist: Dictionary<String, Any> = dictionary[String(describing: self)] as? Dictionary<String, Any> else { throw ErrorCases.NoEntityFound(key: String(describing: self)) }
		guard let directory: String = mnist[directoryKey] as? String else { throw ErrorCases.NoEntityFound(key: directoryKey) }
		guard let items: Dictionary<String, Any> = mnist[set.rawValue] as? Dictionary<String, Any> else { throw ErrorCases.NoEntityFound(key: set.rawValue) }
		guard let labelsDictionary: Dictionary<String, Any> = items[labelKey] as? Dictionary<String, Any> else { throw ErrorCases.NoEntityFound(key: labelKey) }
		guard let imagesDictionary: Dictionary<String, Any> = items[imageKey] as? Dictionary<String, Any> else { throw ErrorCases.NoEntityFound(key: imageKey) }
		guard let labelsCache: String = labelsDictionary[cacheKey] as? String else { throw ErrorCases.NoEntityFound(key: "\(labelKey).\(cacheKey)") }
		guard let labelsFetch: String = labelsDictionary[fetchKey] as? String else { throw ErrorCases.NoEntityFound(key: "\(labelKey).\(fetchKey)") }
		guard let imagesCache: String = imagesDictionary[cacheKey] as? String else { throw ErrorCases.NoEntityFound(key: "\(imageKey).\(cacheKey)") }
		guard let imagesFetch: String = imagesDictionary[fetchKey] as? String else { throw ErrorCases.NoEntityFound(key: "\(imageKey).\(fetchKey)") }
		
		let manager: Manager = try Manager(directory: directory)
		let labels: Data = try manager.gunzip(cache: labelsCache, fetch: labelsFetch)
		let (labelhead, labelbody): (Data, Data) = labels.split(cursor: 2 * MemoryLayout<UInt32>.size)
		let labelheader: Array<Int> = labelhead.toArray().map{Int(UInt32(bigEndian: $0))}
		let label: Array<UInt8> = labelbody.toArray()
		let images: Data = try manager.gunzip(cache: imagesCache, fetch: imagesFetch)
		let (imagehead, imagebody): (Data, Data) = images.split(cursor: 4 * MemoryLayout<UInt32>.size)
		let imageheader: Array<Int> = imagehead.toArray().map{Int(UInt32(bigEndian: $0))}
		guard imageheader.count == 4 else { throw ErrorCases.UnknownError }
		guard imageheader[1] == labelheader[1] else { throw ErrorCases.IncorrectFormat }
		let length: Int = min(imageheader[1], labelheader[1])
		let rows: Int = imageheader[2]
		let cols: Int = imageheader[3]
		let pixel: Array<UInt8> = imagebody.toArray()
		guard length * rows * cols == pixel.count else { throw ErrorCases.IncorrectFormat }
		let image: Array<Array<UInt8>> = pixel.chunk(width: rows * cols)
		return zip(label, image).map {
			MNISTImage(rows: rows, cols: cols, label: Int($0.0), pixel: $0.1)
		}
	}
}
private extension Data {
	func split(cursor: Int) -> (Data, Data){
		return(subdata(in: startIndex..<startIndex.advanced(by: cursor)), subdata(in: startIndex.advanced(by: cursor)..<endIndex))
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
