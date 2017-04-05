//
//  Tanh.metal
//  macOS
//
//  Created by Kota Nakano on 2017/04/04.
//
//

#include <metal_stdlib>
using namespace metal;
constant float W [[ function_constant(0) ]];
constant float3 LRW [[ function_constant(1) ]];
kernel void TanhGenerate(device float * const theta [[ buffer(0) ]],
						 device float const * const phi [[ buffer(1) ]],
						 constant uint const & N [[ buffer(2) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		theta[idx] = W * tanh(p);
	}
}
kernel void TanhGradient(device float * const delta [[ buffer(0) ]],
						 device float const * const theta [[ buffer(1) ]],
						 device float const * const phi [[ buffer(2) ]],
						 constant uint const & N [[ buffer(3) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const f = theta[idx];
		float const g = W * W - f * f;
		delta[idx] = g * dot(LRW, float3(delta[idx], sign(f), f));
	}
}
