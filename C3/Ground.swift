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
	@NSManaged var custom: Int
}
extension Ground {
	var context: Context {
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCase.InvalidContext.description) }
		return context
	}
}
extension Ground {
	public override func awakeFromInsert() {
		super.awakeFromInsert()
		access = Date()
		custom = 0
	}
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		access = Date()
		custom = 0
	}
	public override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		access = Date()
		custom = 0
	}
}
