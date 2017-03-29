//
//  MomentumAdaDelta.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;

constant float rho [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];
template<typename T> T sq(T const x) {
	return x * x;
}
kernel void AdaDeltaOptimize(device float * const theta [[ buffer(0) ]],
							 device float2 * const parameters [[ buffer(1) ]],
							 device const float * const delta [[ buffer(2) ]],
							 constant uint const & N [[ buffer(3) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const g = delta[idx];
		float2 p = parameters[idx];
		p.x = mix(sq(g), p.x, rho);
		float const v = g * sqrt((p.y+epsilon)/(p.x+epsilon));
		p.y = mix(sq(v), p.y, rho);
		theta[idx] -= v;
		parameters[idx] = p;
	}
}
