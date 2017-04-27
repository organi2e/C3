//
//  ViewController.swift
//  Stage
//
//  Created by Kota Nakano on 4/9/17.
//
//

import UIKit
import C3
import Optimizer
import Educator
import Metal

class ViewController: UIViewController {
	
	@IBOutlet weak var bar: UIProgressView?
	@IBOutlet weak var lab: UILabel?
	
	override func viewDidLoad() {
		super.viewDidLoad()
		Thread(block: MNIST(progress: bar, label: lab).run).start()
		// Do any additional setup after loading the view, typically from a nib.
	}
	
	override func didReceiveMemoryWarning() {
		super.didReceiveMemoryWarning()
		// Dispose of any resources that can be recreated.
	}
}

