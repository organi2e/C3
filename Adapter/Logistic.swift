//
//  Logistic.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//

import Metal
public class Logistic: NonLinear {
	public static func adapter(device: MTLDevice) throws -> (Int) -> Adapter {
		let pipeline: (MTLComputePipelineState, MTLComputePipelineState) = try compile(device: device)
		return {
			Logistic(pipeline: pipeline, count: $0)
		}
	}
}
