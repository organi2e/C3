//
//  CorpusTests.swift
//  CorpusTests
//
//  Created by Kota Nakano on 2017/04/10.
//
//

import XCTest
@testable import Corpus

class CorpusTests: XCTestCase {
	func testMNIST() {
		do {
			let x = try MNIST.load(set: .t10k)
			for _ in 0..<16 {
				let img = x[Int(arc4random_uniform(UInt32(x.count)))]
				print(img.label)
				for h in 0..<28 {
					for w in 0..<28 {
						print(img.pixel[h*28+w] > 128 ? 1 : 0, terminator: "")
					}
					print("\r\n", terminator: "")
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
