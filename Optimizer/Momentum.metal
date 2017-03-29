//
//  Momentum.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;

constant float eta [[ function_constant(0) ]];
constant float gamma [[ function_constant(1) ]];

kernel void MomentumOptimize(device float * const theta [[ buffer(0) ]],
							 device float * const parameter [[ buffer(1) ]],
							 device float const * const delta [[ buffer(2) ]],
							 uint const n [[ thread_position_in_grid]]) {
	int const idx = n;
	theta[idx] -= ( parameter[idx] = mix(eta * delta[idx], parameter[idx], gamma));
}
