//
//  Normalizer.metal
//  macOS
//
//  Created by Kota Nakano on 2017/06/09.
//
//

#include <metal_stdlib>
using namespace metal;
constant float gamma [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];
kernel void collect(device float * const target [[ buffer(0) ]],
					device float2 * const parameters [[ buffer(1) ]],
					device float const * const source [[ buffer(2) ]],
					constant uint const & N [[ buffer(3) ]],
					uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float2 const p = parameters[idx];
		float const lambda = rsqrt(fma(p.x, -p.x, p.y));
		target[idx] = select(0.0, lambda, isfinite(lambda)) * ( source[idx] - p.x );
	}
}
kernel void correct(device float * const target [[ buffer(0) ]],
					device float2 const * const parameters [[ buffer(1) ]],
					device float const * const source [[ buffer(2) ]],
					constant uint const & N [[ buffer(3) ]],
					uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float2 const p = parameters[idx];
		float const sigma = sqrt(fma(p.x, -p.x, p.y));
		target[idx] = select(0.0, sigma, isfinite(sigma)) * source[idx];
	}
}
kernel void connect(device float2 * const parameter [[ buffer(0) ]],
					device float * const source [[ buffer(1) ]],
					constant uint const & N [[ buffer(2) ]],
					uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const x = source[idx];
		parameter[idx] = mix(float2(x, x * x), parameter[idx], gamma);
	}
}
