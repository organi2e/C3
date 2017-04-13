//
//  Wikipedia.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/04/12.
//
//

import CoreData

internal class Abstract: NSManagedObject {
	
}
extension Abstract {
	@NSManaged var title: String
	@NSManaged var url: String
	@NSManaged var body: String
}
public class Wikipedia {
	func make(title: String, url: String, body: String) {
	
	}
	func rebuild() throws {
		
		//let url: URL = URL(string: "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-abstract.xml")!
		
	}
	
}
extension Wikipedia {
	private class AbstractParser: NSObject, XMLParserDelegate {
		var stack: Array<String> = Array<String>()
		var title: String = ""
		var body: String = ""
		var url: String = ""
		init(url: URL, handler: @escaping(String, String, String)->Void) throws {
			guard let parser: XMLParser = XMLParser(contentsOf: url) else { throw ErrorCases.UnknownError(message: "xml") }
			super.init()
			parser.delegate = self
			parser.parse()
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
				//print(title, "=", abstract)
				break
			default:
				break
			}
		}
		func parse(url: URL, wikipedia: Wikipedia) {
			wikipedia.complete(title: "", body: "")
		}
	}
	private func complete(title: String, body: String) {
		
	}
	func rebuildAbstract() throws {
		let url: URL = URL(fileURLWithPath: "/tmp/jawiki-latest-abstract.xml")
		let _ = try AbstractParser(url: url, handler: make)
	}
}
