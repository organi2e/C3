//
//  PTB.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/21.
//
//

extension Educator {
	private static let name: String = "PTB"
	private static let plist: String = "plist"
	private static let urlkey: String = "URL"
	public enum PTB: String {
		case train = "TRAIN"
		case valid = "VALID"
		case test = "TEST"
	}
	public func count(ptb: PTB, offset: Int = 0, limit: Int = 0) throws -> Int {
		return try count(domain: type(of: self).name, family: ptb.rawValue, option: Dictionary<String, Any>(), handle: "", offset: offset, limit: limit)
	}
	public func fetch(ptb: PTB, offset: Int = 0, limit: Int = 0) throws -> Array<Corpus> {
		return try fetch(domain: type(of: self).name, family: ptb.rawValue, option: Dictionary<String, Any>(), handle: "", offset: offset, limit: limit)
	}
	public func build(ptb: PTB) throws {
		let name: String = String(describing: type(of: self).name)
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: type(of: self).plist) else {
			throw ErrorCases.NoResourceFound(name: name, extension: type(of: self).plist)
		}
		guard let dictionary: Dictionary<String, String> = try PropertyListSerialization.propertyList(from: Data(contentsOf: plist, options: .mappedIfSafe), options: [], format: nil) as? Dictionary<String, String> else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let path: String = dictionary[type(of: self).urlkey] else {
			throw ErrorCases.NoRecourdFound(name: type(of: self).urlkey)
		}
		guard let url: URL = URL(string: path) else {
			throw ErrorCases.InvalidFormat(of: path, for: URL.self)
		}
		guard let target: String = dictionary[ptb.rawValue] else {
			throw ErrorCases.NoRecourdFound(name: ptb.rawValue)
		}
		var error: Error?
		var store: URL?
		let semaphore: DispatchSemaphore = DispatchSemaphore(value: 0)
		URLSession(configuration: .default).downloadTask(with: URLRequest(url: url, cachePolicy: .returnCacheDataElseLoad)) {
			error = $2
			store = $0
			semaphore.signal()
		}.resume()
		let context: NSManagedObjectContext = NSManagedObjectContext(concurrencyType: .privateQueueConcurrencyType)
		do {
			context.parent = self
		}
		try context.fetch(make(domain: type(of: self).name, family: ptb.rawValue, option: Dictionary<String, Any>(), handle: "", offset: 0, limit: 0)).forEach(context.delete)
		semaphore.wait()
		if let error: Error = error {
			throw error
		}
		guard let file: URL = store else {
			throw ErrorCases.NoFileDownload(from: url)
		}
		try FileHandle(forReadingFrom: file).gunzip().untar {
			if $0.0 == target {
				guard let string: String = String(data: $0.1, encoding: .utf8) else {
					throw ErrorCases.InvalidFormat(of: $0.1, for: String.self)
				}
				let name: String = String(describing: Corpus.self)
				guard let entity: Corpus = NSEntityDescription.insertNewObject(forEntityName: name, into: context) as? Corpus else {
					throw ErrorCases.NoEntityFound(name: name)
				}
				entity.domain = type(of: self).name
				entity.family = ptb.rawValue
				entity.option = Dictionary<String, Any>()
				entity.handle = ptb.rawValue
				
				entity.title = ptb.rawValue
				entity.body = string
			}
		}
		try context.save()
	}
}
