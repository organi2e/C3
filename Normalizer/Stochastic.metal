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
kernel void StochasticAdjust(device float * const du [[ buffer(0) ]],
							 device float * const ds [[ buffer(1) ]],
							 device float2 * const momentum [[ buffer(2) ]],
							 device float4 * const gradient [[ buffer(3) ]],
							 device float const * const u [[ buffer(4) ]],
							 device float const * const s [[ buffer(5) ]],
							 constant uint const & N [[ buffer(6) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float2 const x = float2(u[idx], s[idx]);
		float2 const m = mix(float2(x.x, dot(x, x)), momentum[idx], gamma);
		float4 const g = mix(float4(1, 0, 2 * x), gradient[idx], gamma);
		
		float const v = fma(m.x, -m.x, m.y);
		float const l = 0.5 * log(v) / v;
		float const r = select(0.0, l, isfinite(l));
		
		du[idx] += 2 * m.x * g.x + r * ( g.z - 2 * m.x * g.x );
		ds[idx] +=                 r * ( g.w );
		
		/*
		float const s = sqrt(fma(m.x, -m.x, m.y));
		float const l1 = m.x * g.x;
		float const l2 = ( s - 1 ) / s;
		du[idx] += l1 + l2 * fma(l1, -2, g.z);
		du[idx] +=      l2 * g.w;
		momentum[idx] = m;
		gradient[idx] = g;
		*/
	}
}
