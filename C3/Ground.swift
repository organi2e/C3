//
//  Ground.swift
//  macOS
//
//  Created by Kota Nakano on 4/9/17.
//
//

import CoreData
public class Ground: NSManagedObject {

}
extension Ground {
	@NSManaged var access: Date
}
extension Ground {
	func eval<T>(block: (Context)->T) throws -> T {
		guard let context: Context = managedObjectContext as? Context else {
			throw Context.ErrorCases.InvalidContext
		}
		return block(context)
	}
}
extension Ground {
	public override func awakeFromInsert() {
		super.awakeFromInsert()
		access = Date()
	}
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		access = Date()
	}
}
