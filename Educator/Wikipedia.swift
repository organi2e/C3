//
//  Wikipedia.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/04/12.
//
//

import CoreData

extension Educator {
	private static let name: String = "Wikipedia"
	public enum Wikipedia: String {
		case abstract = "Abstract"
	}
	public func count(wikipedia: Wikipedia, title: String = "", offset: Int = 0, limit: Int = 0) throws -> Int {
		return try count(domain: type(of: self).name, family: wikipedia.rawValue, option: [:], handle: title, offset: offset, limit: limit)
	}
	public func fetch(wikipedia: Wikipedia, title: String = "", offset: Int = 0, limit: Int = 0) throws -> Array<Image> {
		
		return try fetch(domain: type(of: self).name, family: wikipedia.rawValue, option: [:], handle: title, offset: offset, limit: limit)
	}
	public func build(wikipedia: Wikipedia) throws {
		switch wikipedia {
		case .abstract:
			try buildAbstract()
		}
	}
	private class AbstractParserContext: NSManagedObjectContext, XMLParserDelegate {
		var stack: Array<String> = Array<String>()
		var title: String = ""
		var body: String = ""
		var url: String = ""
		let domain: String
		let regexp: NSRegularExpression
		init(name: String, parent context: NSManagedObjectContext) throws {
			domain = name
			regexp = try NSRegularExpression(pattern: "^Wikipedia\\:\\s+(.+)$", options: [])
			super.init(concurrencyType: .privateQueueConcurrencyType)
			parent = context
		}
		func parse(url: URL) throws {
			var error: Error?
			var store: URL?
			let semaphore: DispatchSemaphore = DispatchSemaphore(value: 0)
			URLSession(configuration: .default).downloadTask(with: URLRequest(url: url, cachePolicy: .returnCacheDataElseLoad)) {
				error = $2
				store = $0
				semaphore.signal()
			}.resume()
			guard let parentContext: Educator = parent as? Educator else {
				throw ErrorCases.InvalidFormat(of: parent as Any, for: Educator.self)
			}
			try fetch(parentContext.make(domain: domain, family: Wikipedia.abstract.rawValue, option: Dictionary<String, Any>(), handle: "", offset: 0, limit: 0)).forEach(delete)
			semaphore.wait()
			if let error: Error = error {
				throw error
			}
			guard let file: URL = store else {
				throw ErrorCases.NoFileDownload(from: url)
			}
			guard let parser: XMLParser = XMLParser(contentsOf: file) else { throw ErrorCases.InvalidFormat(of: url, for: "xml") }
			parser.delegate = self
			guard parser.parse() else { throw ErrorCases.InvalidFormat(of: url, for: "xml") }
		}
		required init?(coder aDecoder: NSCoder) {
			assertionFailure("init(coder:) has been not implemented")
			return nil
		}
		private static let doc: String = "doc"
		private static let url: String = "url"
		private static let body: String = "abstract"
		private static let title: String = "title"
		func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes attributeDict: [String : String] = [:]) {
			switch elementName {
			case AbstractParserContext.doc:
				title = ""
				body = ""
				url = ""
			default:
				break
			}
			stack.append(elementName)
		}
		func parser(_ parser: XMLParser, foundCharacters string: String) {
			guard let last: String = stack.last else { parser.abortParsing(); return }
			switch last {
			case AbstractParserContext.title:
				title = title + string
			case AbstractParserContext.body:
				body = body + string
			case AbstractParserContext.url:
				url = url + string
			default:
				break
			}
		}
		func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
			guard let last: String = stack.popLast(), last == elementName else { parser.abortParsing(); return }
			switch last {
			case AbstractParserContext.doc:
				guard let match: NSTextCheckingResult = regexp.firstMatch(in: title, options: [], range: NSRange(location: 0, length: title.characters.count)) else { parser.abortParsing(); break }
				guard 1 < match.numberOfRanges else { parser.abortParsing(); break }
				let range: NSRange = match.rangeAt(1)
				let start: String.Index = title.index(title.startIndex, offsetBy: range.location)
				let end: String.Index = title.index(start, offsetBy: range.length)
				guard let result: Corpus = NSEntityDescription.insertNewObject(forEntityName: String(describing: Corpus.self), into: self) as? Corpus else {
					parser.abortParsing()
					return
				}
				result.domain = domain
				result.family = Wikipedia.abstract.rawValue
				result.option = Dictionary<String, Any>(dictionaryLiteral: ("url", url))
				result.handle = title.substring(with: start..<end)
				result.body = body
			default:
				break
			}
		}
	}
	internal func buildAbstract() throws {
		let name: String = String(describing: type(of: self).name)
		let key: String = Wikipedia.abstract.rawValue
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "plist") else {
			throw ErrorCases.NoPlistFound(name: name)
		}
		guard let dictionary: Dictionary<String, String> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, String> else {
			throw ErrorCases.InvalidFormat(of: plist, for: "plist")
		}
		guard let path: String = dictionary[key] else {
			throw ErrorCases.NoEntityFound(name: key)
		}
		guard let url: URL = URL(string: path) else {
			throw ErrorCases.InvalidFormat(of: path, for: URL.self)
		}
		let context: AbstractParserContext = try AbstractParserContext(name: type(of: self).name, parent: self)
		do {
			context.parent = self
		}
		try context.parse(url: url)
		try context.save()
	}
}

/*
internal class Abstract: NSManagedObject {
	
}
extension Abstract {
	@NSManaged var title: String
	@NSManaged var url: String
	@NSManaged var body: String
}
extension Abstract: Corpus {
	public var name: String {
		return title
	}
}
public class Wikipedia: NSManagedObjectContext {
	public init(storage: URL?) throws {
		super.init(concurrencyType: .privateQueueConcurrencyType)
		guard let model: NSManagedObjectModel = NSManagedObjectModel.mergedModel(from: [Bundle(for: type(of: self))]) else { throw ErrorCases.NoModelFound(name: String(describing: type(of: self))) }
		let store: NSPersistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		let storetype: String = storage == nil ? NSInMemoryStoreType : ["db", "sqlite"].map{ $0 == storage?.pathExtension }.reduce(false, {$0||$1}) ? NSSQLiteStoreType : NSBinaryStoreType
		try store.addPersistentStore(ofType: storetype, configurationName: nil, at: storage, options: nil)
		persistentStoreCoordinator = store
	}
	public required init?(coder aDecoder: NSCoder) {
		assertionFailure("init(coder:) has not been implemented")
		return nil
	}
}
extension Wikipedia {
	public enum Group: String {
		case abstract = "Abstract"
	}
	public func rebuild(group: Group) throws {
		switch group {
		case .abstract:
			try rebuildAbstract()
		}
	}
	public func count(group: Group) throws -> Int {
		return try count(for: NSFetchRequest(entityName: group.rawValue))
	}
	public func fetch(group: Group, name: String = "", offset: Int = 0, count: Int = 0) throws -> Array<Corpus> {
		switch group {
		case .abstract:
			return try fetchAbstract(title: name, offset: offset, count: count)
		}
	}
}
extension Wikipedia {
	public func make<T: NSManagedObject>() throws -> T {
		let name: String = String(describing: T.self)
		var result: T?
		func block() {
			result = NSEntityDescription.insertNewObject(forEntityName: name, into: self) as? T
		}
		performAndWait(block)
		guard let entity: T = result else { throw ErrorCases.NoEntityFound(name: name) }
		return entity
	}
	public func fetch<T: NSManagedObject>(request: NSFetchRequest<T>) throws -> Array<T> {
		var result: Array<T> = []
		var error: Error?
		func block() {
			do {
				result = try fetch(request)
			} catch {
				
			}
		}
		performAndWait(block)
		if let error: Error = error {
			throw error
		}
		return result
	}
}
extension Wikipedia {
	private class AbstractParser: NSObject, XMLParserDelegate {
		var stack: Array<String> = Array<String>()
		var title: String = ""
		var body: String = ""
		var url: String = ""
		let regexp: NSRegularExpression
		let wikipedia: Wikipedia
		init(url: URL, delegate: Wikipedia) throws {
			regexp = try NSRegularExpression(pattern: "^Wikipedia\\:\\s+(.+)$", options: [])
			wikipedia = delegate
			guard let parser: XMLParser = XMLParser(contentsOf: url) else { throw ErrorCases.InvalidFormat(of: url, for: "xml") }
			super.init()
			parser.delegate = self
			guard parser.parse() else { throw ErrorCases.InvalidFormat(of: url, for: "xml") }
		}
		private static let doc: String = "doc"
		private static let url: String = "url"
		private static let body: String = "abstract"
		private static let title: String = "title"
		func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes attributeDict: [String : String] = [:]) {
			switch elementName {
			case AbstractParser.doc:
				title = ""
				body = ""
				url = ""
			default:
				break
			}
			stack.append(elementName)
		}
		func parser(_ parser: XMLParser, foundCharacters string: String) {
			guard let last: String = stack.last else { parser.abortParsing(); return }
			switch last {
			case AbstractParser.title:
				title = title + string
			case AbstractParser.body:
				body = body + string
			case AbstractParser.url:
				url = url + string
			default:
				break
			}
		}
		func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
			guard let last: String = stack.popLast(), last == elementName else { parser.abortParsing(); return }
			switch last {
			case AbstractParser.doc:
				guard let match: NSTextCheckingResult = regexp.firstMatch(in: title, options: [], range: NSRange(location: 0, length: title.characters.count)) else { parser.abortParsing(); break }
				guard 1 < match.numberOfRanges else { parser.abortParsing(); break }
				let range: NSRange = match.rangeAt(1)
				let start: String.Index = title.index(title.startIndex, offsetBy: range.location)
				let end: String.Index = title.index(start, offsetBy: range.length)
				wikipedia.parseAbstract(parser: parser, title: title.substring(with: start..<end), body: body, url: url)
			default:
				break
			}
		}
	}
	private func parseAbstract(parser: XMLParser, title: String, body: String, url: String) {
		guard let result: Abstract = NSEntityDescription.insertNewObject(forEntityName: String(describing: Abstract.self), into: self) as? Abstract else {
			parser.abortParsing()
			return
		}
		result.title = title
		result.body = body
		result.url = url
	}
	internal func rebuildAbstract() throws {
		let name: String = String(describing: type(of: self))
		let key: String = Group.abstract.rawValue
		guard let plist: URL = Bundle(for: type(of: self)).url(forResource: name, withExtension: "plist") else { throw ErrorCases.NoPlistFound(name: name) }
		guard let dictionary: Dictionary<String, String> = try PropertyListSerialization.propertyList(from: try Data(contentsOf: plist), options: [], format: nil) as? Dictionary<String, String> else { throw ErrorCases.InvalidFormat(of: plist, for: "plist") }
		guard let path: String = dictionary[key] else { throw ErrorCases.NoEntityFound(name: key) }
		guard let url: URL = URL(string: path) else { throw ErrorCases.InvalidFormat(of: path, for: URL.self) }
		do {
			undoManager?.beginUndoGrouping()
			defer {
				undoManager?.endUndoGrouping()
			}
			try fetch(request: NSFetchRequest<Abstract>(entityName: String(describing: Abstract.self))).forEach(delete)
			let _ = try AbstractParser(url: url, delegate: self)
			try save()
		} catch {
			undoManager?.undo()
		}
	}
	internal func fetchAbstract(title: String, offset: Int, count: Int) throws -> Array<Corpus> {
		let name: String = String(describing: Abstract.self)
		let request: NSFetchRequest<Abstract> = NSFetchRequest<Abstract>(entityName: name)
		if !title.isEmpty {
			request.predicate = NSPredicate(format: "title = %@", title)
		}
		request.fetchOffset = offset
		request.fetchLimit = count
		return try fetch(request)
	}
}
*/
