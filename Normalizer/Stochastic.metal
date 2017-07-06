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
		float2 const m = mix(momentum[idx], float2(x.x, dot(x, x)), gamma);
//		float2 const m = float2(gamma * momentum[idx].x + ( 1 - gamma ) * x.x,
//								gamma * momentum[idx].y + ( 1 - gamma ) * x.y);
//		float4 const g = mix(float4(1, 0, 2 * x.x, 2 * x.y), gradient[idx], gamma);
		float4 const g = gamma * float4(1,
										0,
										2 * x.x,
										2 * x.y);// + gamma * float4(0,
//																		  0,
//																		  gradient[idx].z,
//																		  gradient[idx].w);
		
		float const v = fma(m.x, -m.x, m.y);
		float const l = 0.5 * log(v) / v;
		float const r = select(0.0, l, isfinite(l));
		float const n = 2 * m.x * g.x;
		
		du[idx] += n + r * ( g.z - n );
		ds[idx] +=     r * ( g.w );
		
		momentum[idx] = m;
		gradient[idx] = g;
	}
}
