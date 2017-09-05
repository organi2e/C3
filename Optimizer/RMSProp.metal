//
//  RMSProp.metal
//  macOS
//
//  Created by Kota Nakano on 2017/08/29.
//
//

#include <metal_stdlib>
using namespace metal;

constant float3 alpha [[ function_constant(0) ]];
constant float gamma [[ function_constant(1) ]];
constant float epsilon [[ function_constant(2) ]];

template<typename T> T sq(T const x) {
	return x * x;
}
kernel void RMSPropOptimize(device float * const theta [[ buffer(0) ]],
														device float * const parameters [[ buffer(1) ]],
														device const float * const delta [[ buffer(2) ]],
														constant uint const & N [[ buffer(3) ]],
														uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const g = delta[idx];
		float const p = mix(sq(g), parameters[idx], gamma);
		
		float const r = rsqrt(p+epsilon);//approx of 1/(sqrt(p)+eps)
		float const t = theta[idx];
		theta[idx] -= dot(alpha, float3(t, sign(t), g * select(0.0, r, isnormal(r))));
		
		parameters[idx] = p;
	}
}
