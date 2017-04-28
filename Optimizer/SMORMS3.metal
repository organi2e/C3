//
//  SMORMS3.metal
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

#include <metal_stdlib>
using namespace metal;

constant float3 alpha [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];

kernel void SMORMS3Optimize(device float * const theta [[ buffer(0) ]],
							device float * const parameters [[ buffer(1) ]],
							device float * const delta [[ buffer(2) ]],
							constant uint const & N [[ buffer(3) ]],
							uint const n [[ thread_position_in_grid ]]) {
	
	if ( n < N ) {
		
		int const idx = n;
		
		//fetch
		float3 p = ((device float3*)parameters)[idx];
		float const g = delta[idx];
		
		p.xy = mix(p.xy, float2(g*g, g), 1/(1+p.z));
//		p.x = ( 1 - r ) * p.x + r * g;
//		p.y = ( 1 - r ) * p.y + r * g * g;
		
		float const s = rsqrt(p.x+epsilon);
		float const r = select(0.0, s, isnormal(s));//Avoid epsilon
		float const u = p.y * r;
		float const x = u * u;
		float const t = theta[idx];
		p.z = fma(p.z, 1 - x, 1);
		
		//
		//float const u = (p.y = mix(g, p.y, b)) * t;
		//float const x = abs(u);
		//float const x = u * u;
		//p.x = fma(p.x, 1 - x, 1);
		
		//update
		//theta[idx] -= alpha * x * t * g;//or min(alpha, v)
		//theta[idx] -= fma(, dot(alpha.xy, ));
		theta[idx] -= fma(g*r, min(x, alpha.z), dot(alpha.xy, float2(t, sign(t))));
		//g * min(alpha.z, x) * r;
		//theta[idx] -= g * min(alpha.z, x) * r;
		((device float3*)parameters)[idx] = p;
	}
}
