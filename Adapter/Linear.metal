//
//  Linear.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/03/10.
//
//

#include <metal_stdlib>
using namespace metal;
constant float L1 [[ function_constant(0) ]];
constant float L2 [[ function_constant(1) ]];
kernel void LinearGradient(device float * const delta [[ buffer(0) ]],
						   device float const * const theta [[ buffer(1) ]],
						   device float const * const phi [[ buffer(2) ]],
						   constant uint const & N [[ buffer(3) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const t = theta[idx];
		delta[idx] += dot(float2(L1, L2), float2(sign(t), t));
	}
}
