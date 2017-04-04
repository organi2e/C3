//
//  Floor.metal
//  macOS
//
//  Created by Kota Nakano on 2017/04/04.
//
//

#include <metal_stdlib>
using namespace metal;
constant float3 LRW [[ function_constant(0) ]];
kernel void FloorGenerate(device float * const theta [[ buffer(0) ]],
								device float const * const phi [[ buffer(1) ]],
								constant uint const & N [[ buffer(2) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		theta[idx] = select(exp(p-1), p, 1<p);
	}
}
kernel void FloorGradient(device float * const delta [[ buffer(0) ]],
								device float const * const theta [[ buffer(1) ]],
								device float const * const phi [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		float const v = exp(p-1);
		float const f = select(exp(v-1), p, 1<p);
		float const g = select(exp(v-1), 1.0, 1<p);
		delta[idx] = g * dot(LRW, float3(delta[idx], sign(f), f));
	}
}
