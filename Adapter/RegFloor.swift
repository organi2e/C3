//
//  RegFloor.swift
//  tvOS
//
//  Created by Kota Nakano on 4/8/17.
//
//

import Metal
public class RegFloor: NonLinear {
	public static func adapter(device: MTLDevice) throws -> (Int) -> Adapter {
		let pipeline: (MTLComputePipelineState, MTLComputePipelineState) = try compile(device: device)
		return {
			Regular(pipeline: pipeline, count: $0)
		}
	}
}
