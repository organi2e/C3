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
	var context: Context {
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCases.InvalidContext.description) }
		return context
	}
}
extension Ground {
	public override func awakeFromInsert() {
		super.awakeFromInsert()
		access = Date()
	}
}
