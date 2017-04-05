//
//  Logistic.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/04/05.
//
//

#include <metal_stdlib>
using namespace metal;
constant float3 LRW [[ function_constant(0) ]];
kernel void LogisticGenerate(device float * const theta [[ buffer(0) ]],
								device float const * const phi [[ buffer(1) ]],
								constant uint const & N [[ buffer(2) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		theta[idx] = 1/(1+exp(-p));
	}
}
kernel void LogisticGradient(device float * const delta [[ buffer(0) ]],
								device float const * const theta [[ buffer(1) ]],
								device float const * const phi [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const f = theta[idx];
		float const g = f*(1-f);
		delta[idx] = g * dot(LRW, float3(delta[idx], sign(f), f));
	}
}
