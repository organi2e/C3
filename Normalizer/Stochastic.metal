//
//  StochasticNormalizer.metal
//  macOS
//
//  Created by Kota Nakano on 2017/06/09.
//
//

#include <metal_stdlib>
using namespace metal;
constant float gamma [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];
kernel void StochasticCollect(device float * const target [[ buffer(0) ]],
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
kernel void StochasticCorrect(device float * const target [[ buffer(0) ]],
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
kernel void StochasticAverage(device float4 * const parameters [[ buffer(0) ]],
							  device float const * const mu [[ buffer(1) ]],
							  device float const * const sigma [[ buffer(2) ]],
							  constant uint const & N [[ buffer(3) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float4 p = parameters[idx];
		float const u = mu[idx];
		float const s = sigma[idx];
//		float2 const x = float2(mu[idx], sigma[idx]);
		p.x = p.x * gamma + ( 1 - gamma ) * u;
		p.y = p.y * gamma + ( 1 - gamma ) * ( u * u + s * s );
		p.z = p.z * gamma + ( 1 - gamma ) * 2 * u;
		p.w = p.w * gamma + ( 1 - gamma ) * 2 * s;
		parameters[idx] = p;
	}
}
kernel void StochasticScaling(device float * const target_mu [[ buffer(0) ]],
							  device float * const target_sigma [[ buffer(1) ]],
							  device float const * const source_mu [[ buffer(2) ]],
							  device float const * const source_sigma [[ buffer(3) ]],
							  device float4 const * const parameters [[ buffer(4) ]],
							  constant uint const & N [[ buffer(5) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float4 const p = parameters[idx];
		float const u = p.x;
		float const v = p.y - p.x * p.x;
//		float const s = sqrt(p.y - p.x * p.x);
		target_mu[idx] += ( 1 - gamma ) * u + log( v ) * p.z / v;
		target_sigma[idx] += log( v ) * p.w / v;
	}
}
kernel void StochasticConnect(device float2 * const parameter [[ buffer(0) ]],
							  device float * const source [[ buffer(1) ]],
							  constant uint const & N [[ buffer(2) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const x = source[idx];
		parameter[idx] = mix(float2(x, x * x), parameter[idx], gamma);
	}
}
