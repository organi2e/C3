//
//  StochasticGradientDescent.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

#include <metal_stdlib>
using namespace metal;

constant float3 eta [[ function_constant(0) ]];
kernel void StochasticGradientDescentOptimize(device float * const theta [[ buffer(0) ]],
											  device float const * const delta [[ buffer(1) ]],
											  constant uint & N [[ buffer(2) ]],
											  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const t = theta[idx];
		theta[idx] -= dot(eta, float3(delta[idx], t, sign(t)));
	}
}
