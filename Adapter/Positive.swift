//
//  Positive.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/29.
//
//

import Metal
public class Positive: NonLinear {
	public static func adapter(device: MTLDevice) throws -> (Int) -> Adapter {
		let pipeline: (MTLComputePipelineState, MTLComputePipelineState) = try compile(device: device)
		return {
			Positive(pipeline: pipeline, count: $0)
		}
	}
}
