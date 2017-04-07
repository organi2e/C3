//
//  RegFloor.metal
//  tvOS
//
//  Created by Kota Nakano on 4/8/17.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void RegFloorGenerate(device float * const theta [[ buffer(0) ]],
							device float const * const phi [[ buffer(1) ]],
							constant uint const & N [[ buffer(2) ]],
							uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		theta[idx] = select(exp(p), log(1+p)+1, p>0);
	}
}
kernel void RegFloorGradient(device float * const delta [[ buffer(0) ]],
							device float const * const theta [[ buffer(1) ]],
							device float const * const phi [[ buffer(2) ]],
							constant uint const & N [[ buffer(3) ]],
							uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = phi[idx];
		delta[idx] *= select(exp(p), 1/(1+p), p>0);
	}
}
