//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import Cocoa

class ViewController: NSViewController {

	override func viewDidLoad() {
		super.viewDidLoad()

		// Do any additional setup after loading the view.
	}

	override var representedObject: Any? {
		didSet {
		// Update the view, if already loaded.
		}
	}
	

}
extension Data {
	func split(cursor: Int) -> (Data, Data) {
		let m: Data.Index = startIndex.advanced(by: count)
		return (subdata(in: startIndex..<m), subdata(in: m..<endIndex))
	}
	func toArray<T>() -> Array<T> {
		return withUnsafeBytes { Array<T>(UnsafeBufferPointer<T>(start: $0, count: count / MemoryLayout<T>.size)) }
	}
}
extension Array {
	func chunk(count: Int) -> [[ Element ]] {
		return stride(from: 0, to: count, by: count).map {
			Array(self[$0..<$0.advanced(by: count)])
		}
	}
}
