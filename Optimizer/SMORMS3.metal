//
//  SMORMS3.metal
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

#include <metal_stdlib>
using namespace metal;

//constant float2 SMORMS3Parameter [[ function_constant(0) ]];//not used on this customized version

constant float alpha [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];

kernel void SMORMS3Optimize(device float * const theta [[ buffer(0) ]],
							device float3 * const parameters [[ buffer(1) ]],
							device float * const delta [[ buffer(2) ]],
							constant uint const & N [[ buffer(3) ]],
							uint const n [[ thread_position_in_grid ]]) {
	
	if ( n < N ) {
		
		int const idx = n;
		
		//fetch
		float3 p = parameters[idx];
		float const g = delta[idx];
		
		float const r = 1 / ( 1 + p.z );
		p.xy = mix(p.xy, float2(g*g, g), r);
//		p.x = ( 1 - r ) * p.x + r * g;
//		p.y = ( 1 - r ) * p.y + r * g * g;
		
		float const s = rsqrt(p.x+epsilon);
		float const t = select(0.0, s, isnormal(s));//Avoid epsilon
		float const u = p.y * t;
		float const x = u * u;
		p.z = fma(p.z, 1 - x, 1);

		
		//
		//float const u = (p.y = mix(g, p.y, b)) * t;
		//float const x = abs(u);
		//float const x = u * u;
		//p.x = fma(p.x, 1 - x, 1);
		
		//update
		//theta[idx] -= alpha * x * t * g;//or min(alpha, v)
		theta[idx] -= g * min(alpha, x) * t;
		parameters[idx] = p;
	}
}
