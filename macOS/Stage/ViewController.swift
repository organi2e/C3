//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import Cocoa
import Metal

import C3
import Optimizer
import Educator

class ViewController: NSViewController {

	@IBOutlet var progress: NSProgressIndicator?
	@IBOutlet var label: NSTextField?
	
	override func viewDidLoad() {
		super.viewDidLoad()
		Thread(block: MNIST(progress: progress, label: label).gan).start()
	}
	override var representedObject: Any? {
		didSet {
		// Update the view, if already loaded.
		}
	}
	

}
