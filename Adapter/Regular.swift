//
//  Regular.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public class Regular: NonLinear {
	public static func adapter(device: MTLDevice) throws -> (Int) -> Adapter {
		let pipeline: (MTLComputePipelineState, MTLComputePipelineState) = try compile(device: device)
		return {
			Regular(pipeline: pipeline, count: $0)
		}
	}
}
