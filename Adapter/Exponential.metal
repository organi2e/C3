//
//  Exponential.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void ExponentialGenerate(device float * const theta [[ buffer(0) ]],
								device float const * const phi [[ buffer(1) ]],
								constant uint const & N [[ buffer(2) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const v = phi[idx];
		theta[idx] = select(exp(v-1), v, 1<v);
	}
}
kernel void ExponentialGradient(device float * const delta [[ buffer(0) ]],
								device float const * const theta [[ buffer(1) ]],
								device float const * const phi [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const v = phi[idx];
		delta[idx] = select(exp(v-1), 1.0, 1<v) * delta[idx] + 1e-3 * theta[idx];
	}
}
kernel void ExponentialAdapt(device float * const phi [[ buffer(0) ]],
								device float const * const theta [[ buffer(1) ]],
								device float const * const delta [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		phi[idx] -= theta[idx] * delta[idx];
	}
}
