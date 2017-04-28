//
//  Softplus.metal
//  macOS
//
//  Created by Kota Nakano on 2017/04/04.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void SoftplusGenerate(device float * const theta [[ buffer(0) ]],
							 device float const * const phi [[ buffer(1) ]],
							 constant uint const & N [[ buffer(2) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		theta[idx] = log ( 1 + exp ( p ) );
	}
}
kernel void SoftplusGradient(device float * const delta [[ buffer(0) ]],
							 device float const * const theta [[ buffer(1) ]],
							 device float const * const phi [[ buffer(2) ]],
							 constant uint const & N [[ buffer(3) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		delta[idx] /= 1 + exp(-phi[idx]);
	}
}
