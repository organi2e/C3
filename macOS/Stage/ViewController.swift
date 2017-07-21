//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import Cocoa

class ViewController: NSViewController {

	@IBOutlet var progress: NSProgressIndicator?
	@IBOutlet var label: NSTextField?
	
	override func viewDidLoad() {
		super.viewDidLoad()
//		try!Thread(block: MNIST(progress: progress, label: label).gan).start()
		try!Thread(block: MNIST(progress: progress, label: label).semisupervised).start()
	}
	override var representedObject: Any? {
		didSet {
		// Update the view, if already loaded.
		}
	}
	

}
