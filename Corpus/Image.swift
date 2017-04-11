//
//  Image.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/11.
//
//

import Accelerate
public protocol Image {
	var rows: Int { get }
	var cols: Int { get }
	var label: Int { get }
	var pixel: Array<UInt8> { get }
	var vImage: vImage_Buffer { get }
	var channel: Int { get }
}
extension Image {
	public func save(to: URL) throws {
		print(rows)
	}
}
