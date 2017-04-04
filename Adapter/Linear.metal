//
//  Linear.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/03/10.
//
//

#include <metal_stdlib>
using namespace metal;
constant float3 LRW [[ function_constant(0) ]];
kernel void LinearGradient(device float * const delta [[ buffer(0) ]],
						   device float const * const theta [[ buffer(1) ]],
						   device float const * const phi [[ buffer(2) ]],
						   constant uint const & N [[ buffer(3) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = theta[idx];
		delta[idx] = dot(LRW, float3(delta[idx], sign(p), p));
	}
}
