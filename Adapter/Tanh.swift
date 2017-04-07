//
//  Tanh.swift
//  macOS
//
//  Created by Kota Nakano on 2017/04/04.
//
//

import Metal
public class Tanh: NonLinear {
	public static func adapter(device: MTLDevice) throws -> (Int) -> Adapter {
		let pipeline: (MTLComputePipelineState, MTLComputePipelineState) = try compile(device: device)
		return {
			Tanh(pipeline: pipeline, count: $0)
		}
	}
}
