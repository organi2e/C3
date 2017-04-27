//
//  Cauchy.metal
//  macOS
//
//  Created by Kota Nakano on 2017/04/26.
//
//

#include <metal_stdlib>
using namespace metal;
constant int4 const M_INC = int4(0, 1, 2, 3);
inline float4x4 abs(const float4x4 x) {
	return float4x4(abs(x[0]),
					abs(x[1]),
					abs(x[2]),
					abs(x[3]));
}
/*----------------------------------------------------------------*/
kernel void CauchyCollectX(device float * const m [[ buffer(0) ]],
						   device float * const v [[ buffer(1) ]],
						   device float const * const w [[ buffer(2) ]],
						   device float const * const u [[ buffer(3) ]],
						   device float const * const s [[ buffer(4) ]],
						   constant uint2 & S [[ buffer(5) ]],
						   threadgroup float2x4 * shared [[ threadgroup(0) ]],
						   uint const t [[ thread_position_in_threadgroup ]],
						   uint const T [[ threads_per_threadgroup ]],
						   uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float2x4 value = float2x4(0);
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		/*
		 for ( int r = 0 ; r < 4 ; ++ r ) {
			for ( int c = 0 ; c < 4 ; ++ c ) {
		 um[r][c] = rows_mask[r] && cols_mask[c] ? w_mu[row[r]*size.y+col[c]] : 0;
		 sm[r][c] = rows_mask[r] && cols_mask[c] ? w_sigma[row[r]*size.y+col[c]] : 0;
			}
			f[r] = cols_mask[r] ? x[col[r]] : 0;
			sm[r] *= sm[r];
		 }
		 */
		int4 const idx = row * size.y + k;
		float4x4 const x = float4x4(select(0, *(device float4*)(w+idx.x), rows_mask.x && cols_mask),
									select(0, *(device float4*)(w+idx.y), rows_mask.y && cols_mask),
									select(0, *(device float4*)(w+idx.z), rows_mask.z && cols_mask),
									select(0, *(device float4*)(w+idx.w), rows_mask.w && cols_mask));
		value += float2x4(    select(0, *(device float4*)(u+k), cols_mask)  *     x,
						  abs(select(0, *(device float4*)(s+k), cols_mask)) * abs(x));
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float2x4 * accum = shared + a;
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(m+row.x) += (*accum)[0].xyzw;
		*(device float4*)(v+row.x) += (*accum)[1].xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(m+row.x) += (*accum)[0].xyz;
		*(device float3*)(v+row.x) += (*accum)[1].xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(m+row.x) += (*accum)[0].xy;
		*(device float2*)(v+row.x) += (*accum)[1].xy;
	} else if ( rows_mask.x ) {
		*(device float *)(m+row.x) += (*accum)[0].x;
		*(device float *)(v+row.x) += (*accum)[1].x;
	}
}
kernel void CauchyCollectW(device float * const m [[ buffer(0) ]],
						   device float * const v [[ buffer(1) ]],
						   device float const * const u [[ buffer(2) ]],
						   device float const * const s [[ buffer(3) ]],
						   device float const * const x [[ buffer(4) ]],
						   constant uint2 & S [[ buffer(5) ]],
						   threadgroup float2x4 * shared [[ threadgroup(0) ]],
						   uint const t [[ thread_position_in_threadgroup ]],
						   uint const T [[ threads_per_threadgroup ]],
						   uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float2x4 value = float2x4(0);
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		/*
		 for ( int r = 0 ; r < 4 ; ++ r ) {
			for ( int c = 0 ; c < 4 ; ++ c ) {
		 um[r][c] = rows_mask[r] && cols_mask[c] ? w_mu[row[r]*size.y+col[c]] : 0;
		 sm[r][c] = rows_mask[r] && cols_mask[c] ? w_sigma[row[r]*size.y+col[c]] : 0;
			}
			f[r] = cols_mask[r] ? x[col[r]] : 0;
			sm[r] *= sm[r];
		 }
		 */
		int4 const idx = row * size.y + k;
		
		float4 const f = select(0, *(device float4*)(x + k), cols_mask);
		
		value += float2x4(    f *      float4x4(select(0, *(device float4*)(u+idx.x), rows_mask.x && cols_mask),
												select(0, *(device float4*)(u+idx.y), rows_mask.y && cols_mask),
												select(0, *(device float4*)(u+idx.z), rows_mask.z && cols_mask),
												select(0, *(device float4*)(u+idx.w), rows_mask.w && cols_mask)),
						  abs(f) * abs(float4x4(select(0, *(device float4*)(s+idx.x), rows_mask.x && cols_mask),
												select(0, *(device float4*)(s+idx.y), rows_mask.y && cols_mask),
												select(0, *(device float4*)(s+idx.z), rows_mask.z && cols_mask),
												select(0, *(device float4*)(s+idx.w), rows_mask.w && cols_mask))));
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float2x4 * accum = shared + a;
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(m+row.x) += (*accum)[0].xyzw;
		*(device float4*)(v+row.x) += (*accum)[1].xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(m+row.x) += (*accum)[0].xyz;
		*(device float3*)(v+row.x) += (*accum)[1].xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(m+row.x) += (*accum)[0].xy;
		*(device float2*)(v+row.x) += (*accum)[1].xy;
	} else if ( rows_mask.x ) {
		*(device float *)(m+row.x) += (*accum)[0].x;
		*(device float *)(v+row.x) += (*accum)[1].x;
	}
}
kernel void CauchyCollectC(device float * const m [[ buffer(0) ]],
						   device float * const v [[ buffer(1) ]],
						   device float const * const u [[ buffer(2) ]],
						   device float const * const s [[ buffer(3) ]],
						   constant uint const & N [[ buffer(4) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		m[idx] +=     u[idx];
		v[idx] += abs(s[idx]);
	}
}
kernel void CauchyCollectD(device float * const m [[ buffer(0) ]],
						   device float * const v [[ buffer(1) ]],
						   device float const * const d [[ buffer(2) ]],
						   device float const * const u [[ buffer(3) ]],
						   device float const * const s [[ buffer(4) ]],
						   constant uint const & N [[ buffer(5) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const r = d[idx];
		m[idx] +=     r*u[idx];
		v[idx] += abs(r*s[idx]);
	}
}
kernel void CauchyCollectF(device float * const u [[ buffer(0) ]],
						   device float * const s [[ buffer(1) ]],
						   constant uint const & N [[ buffer(2) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		s[idx] = abs(s[idx]);
	}
}
/*----------------------------------------------------------------*/
kernel void CauchyCorrectJ(device float * const dx [[ buffer(0) ]],
						   device float const * const ju [[ buffer(1) ]],
						   device float const * const js [[ buffer(2) ]],
						   device float const * const gu [[ buffer(3) ]],
						   device float const * const gs [[ buffer(4) ]],
						   constant uint2 const & S [[ buffer(5) ]],
						   threadgroup float4 * shared [[ threadgroup(0) ]],
						   uint const t [[ thread_position_in_threadgroup ]],
						   uint const T [[ threads_per_threadgroup ]],
						   uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 value = 0;
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = col * size.x + row.x;
		
		value +=
		float4x4(select(0, *(device float4*)(ju+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(ju+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(ju+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(ju+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(gu+k), cols_mask)
		+
		float4x4(select(0, *(device float4*)(js+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(js+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(js+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(js+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(gs+k), cols_mask);
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(dx+row.x) += accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(dx+row.x) += accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(dx+row.x) += accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(dx+row.x) += accum->x;
	}
}
kernel void CauchyCorrectG(device float * const dx [[ buffer(0) ]],
						   device float const * const x [[ buffer(1) ]],
						   device float const * const d [[ buffer(2) ]],
						   constant uint const & N [[ buffer(3) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += (x[idx] - d[idx]);
	}
}
kernel void CauchyCorrectN(device float * const dx [[ buffer(0) ]],
						   device float const * const x [[ buffer(1) ]],
						   constant uint const & N [[ buffer(2) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += 2 * x[idx] - 1;
	}
}
kernel void CauchyCorrectP(device float * const dx [[ buffer(0) ]],
						   device float const * const u [[ buffer(1) ]],
						   device float const * const s [[ buffer(2) ]],
						   device float const * const d [[ buffer(3) ]],
						   constant uint const & N [[ buffer(4) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = fma(atan(u[idx]/s[idx]), M_1_PI_F, 0.5);
		dx[idx] += ( p - d[idx] ) / p / ( 1 - p );
	}
}
kernel void CauchyCorrectV(device float * const dx [[ buffer(0) ]],
						   device float const * const u [[ buffer(1) ]],
						   device float const * const s [[ buffer(2) ]],
						   device float const * const d [[ buffer(3) ]],
						   constant uint const & N [[ buffer(4) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += u[idx] - d[idx];
	}
}
/*----------------------------------------------------------------*/
kernel void CauchyJacobianX(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const x [[ buffer(2) ]],
							device float const * const u [[ buffer(3) ]],
							device float const * const s [[ buffer(4) ]],
							constant uint2 const & N [[ buffer(5) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		float const v = x[cols];
		ju[idx] +=     u[idx];
		js[idx] += abs(s[idx]) * sign(v);
	}
}
kernel void CauchyJacobianA(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const u [[ buffer(2) ]],
							device float const * const s [[ buffer(3) ]],
							device float const * const x [[ buffer(4) ]],
							constant uint2 const & N [[ buffer(5) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		float const v = x[cols];
		ju[idx] +=                    v;
		js[idx] += sign(s[idx]) * abs(v);
	}
}
kernel void CauchyJacobianB(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const Bu [[ buffer(2) ]],
							device float const * const Bs [[ buffer(3) ]],
							device float const * const Y [[ buffer(4) ]],
							device float const * const Ju [[ buffer(5) ]],
							device float const * const Js [[ buffer(6) ]],
							device float const * const Pu [[ buffer(7) ]],
							device float const * const Ps [[ buffer(8) ]],
							constant uint4 const & mnkl [[ buffer(9) ]],
							threadgroup float4x4 * const sharedB [[ threadgroup(0) ]],
							threadgroup float4x4 * const sharedP [[ threadgroup(1) ]],
							uint2 const t [[ thread_position_in_threadgroup ]],
							uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	int const tx = t.x;
	int const ty = t.y;
	threadgroup float4x4 * const cacheB = sharedB + tx * L;
	threadgroup float4x4 * const cacheP = sharedP + ty * L;
	thread float4x4 ru(0), rs(0), rb, rp;
	int2 const b = 4 * int2( g * L + t );
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		float4 const gu = *(device float4*)(Ju+p.y);
		for ( int3 idx = int3(int2(b.x, p.x) * int2(K, N) + int2(p.y, b.y), 0), dx = int3(K, N, 1) ; idx.z < 4 ; idx += dx ) {
			bool4 const bmask = b.x + idx.z < M && p.y + M_INC < K;
			bool4 const pmask = p.x + idx.z < K && b.y + M_INC < N;
			rb[idx.z] = select(0, *(device float4*)(Bu+idx.x)*gu, bmask);
			rp[idx.z] = select(0, *(device float4*)(Pu+idx.y), pmask);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rb;
		cacheP[tx] = rp;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			rb = cacheB[l], rp = cacheP[l];
			ru[0] += rp * rb[0];
			ru[1] += rp * rb[1];
			ru[2] += rp * rb[2];
			ru[3] += rp * rb[3];
		}
		float4 const gs = *(device float4*)(Js+p.y)*sign(*(device float4*)(Y+p.y));
		for ( int3 idx = int3(int2(b.x, p.x) * int2(K, N) + int2(p.y, b.y), 0), dx = int3(K, N, 1) ; idx.z < 4 ; idx += dx ) {
			bool4 const bmask = b.x + idx.z < M && p.y + M_INC < K;
			bool4 const pmask = p.x + idx.z < K && b.y + M_INC < N;
			rb[idx.z] = select(0, abs(*(device float4*)(Bs+idx.x))*gs, bmask);
			rp[idx.z] = select(0,     *(device float4*)(Ps+idx.y), pmask);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rb;
		cacheP[tx] = rp;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			rb = cacheB[l], rp = cacheP[l];
			rs[0] += rp * rb[0];
			rs[1] += rp * rb[1];
			rs[2] += rp * rb[2];
			rs[3] += rp * rb[3];
		}
	}
	for ( int2 row = int2(0, b.x), rows = int2(4, M) ; all(row < rows) ; ++ row ) {
		for ( int2 col = int2(0, b.y), cols = int2(4, N) ; all(col < cols) ; ++ col ) {
			ju [ row.y * N + col.y ] += ru [ row.x ] [ col.x ];
			js [ row.y * N + col.y ] += rs [ row.x ] [ col.x ];
		}
	}
}
kernel void CauchyJacobianC(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const u [[ buffer(2) ]],
							device float const * const s [[ buffer(3) ]],
							constant uint2 const & N [[ buffer(4) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		ju[idx] += 1;
		js[idx] += sign(s[rows]);
	}
}
kernel void CauchyJacobianD(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const d [[ buffer(2) ]],
							device float const * const u [[ buffer(3) ]],
							device float const * const s [[ buffer(4) ]],
							constant uint2 const & N [[ buffer(5) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		ju[idx] +=     u[rows];
		js[idx] += abs(s[rows]) * sign(d[rows]);
	}
}
kernel void CauchyJacobianE(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const u [[ buffer(2) ]],
							device float const * const s [[ buffer(3) ]],
							device float const * const d [[ buffer(4) ]],
							device float const * const pu [[ buffer(5) ]],
							device float const * const ps [[ buffer(6) ]],
							constant uint2 const & N [[ buffer(7) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		float const r = d[rows];
		ju[idx] +=     r                 * pu[idx];
		js[idx] += abs(r) * sign(s[idx]) * ps[idx];
	}
}
kernel void CauchyJacobianF(device float * const ju [[ buffer(0) ]],
							device float * const js [[ buffer(1) ]],
							device float const * const u [[ buffer(2) ]],
							device float const * const s [[ buffer(3) ]],
							constant uint2 const & N [[ buffer(4) ]],
							uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		js[idx] /= s[rows];
	}
}
/*----------------------------------------------------------------*/
constant uint3 xorshift16 [[ function_constant(0) ]];
kernel void CauchyActivateP(device float * const f [[ buffer(0) ]],
							device float * const gu [[ buffer(1) ]],
							device float * const gs [[ buffer(2) ]],
							device float const * const u [[ buffer(3) ]],
							device float const * const s [[ buffer(4) ]],
							constant ushort const * const seeds [[ buffer(5) ]],
							constant uint const & N [[ buffer(6) ]],
							uint const t [[ thread_position_in_threadgroup ]],
							uint const T [[ threadgroups_per_grid ]]) {
	//	ushort4 seq = *(constant ushort4*)(seeds+4*t);
	for ( int k = 4 * t, K = N, dk = 4 * T ; k < K ; k += dk ) {
		float4 const m = *(device float4*)(u+k);
		float4 const v = *(device float4*)(s+k);
		float4 const r = 1 / ( m * m + v * v );
		//		float4 const y = step(float4(seq), fma(erf(M_SQRT1_2_F*x), 32767, 32768));
		float4 const y = fma(atan(m/v), M_1_PI_F, 0.5);
		float4 const ju = M_1_PI_F * v * r;
		float4 const js = M_1_PI_F *-m * r;
		//		seq ^= seq << xorshift16.x;
		//		seq ^= seq >> xorshift16.y;
		//		seq ^= seq << xorshift16.z;
		switch(min(4, K-k)) {
			case 4:
				*(device float4*)(f+k) = y.xyzw;
				*(device float4*)(gu+k) = ju.xyzw;
				*(device float4*)(gs+k) = js.xyzw;
				break;
			case 3:
				*(device float3*)(f+k) = y.xyz;
				*(device float3*)(gu+k) = ju.xyz;
				*(device float3*)(gs+k) = js.xyz;
				break;
			case 2:
				*(device float2*)(f+k) = y.xy;
				*(device float2*)(gu+k) = ju.xy;
				*(device float2*)(gs+k) = js.xy;
				break;
			case 1:
				*(device float *)(f+k) = y.x;
				*(device float *)(gu+k) = ju.x;
				*(device float *)(gs+k) = js.x;
				break;
		}
	}
}
kernel void CauchyDerivateP(device float * const du [[ buffer(0) ]],
							device float * const ds [[ buffer(1) ]],
							device float const * const f [[ buffer(2) ]],
							device float const * const gu [[ buffer(3) ]],
							device float const * const gs [[ buffer(4) ]],
							device float const * const u [[ buffer(5) ]],
							device float const * const s [[ buffer(6) ]],
							constant uint const & N [[ buffer(7) ]],
							uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		//		float const p = fma(erf(M_SQRT1_2_F*u[idx]/s[idx]), 32767.0/65536.0, 0.5);
		//		float const d = sign(du[idx]);//p - saturate(p - sign(du[idx]));
		float const g = du[idx];// / p / ( 1 - p );//p - saturate(p - sign(du[idx]));//d / p / ( 1 - p );
		du[idx] = g * gu[idx];
		ds[idx] = g * gs[idx];
	}
}
kernel void CauchyActivateV(device float * const f [[ buffer(0) ]],
							device float * const gu [[ buffer(1) ]],
							device float * const gs [[ buffer(2) ]],
							device float const * const u [[ buffer(3) ]],
							device float const * const s [[ buffer(4) ]],
							constant ushort const * const seeds [[ buffer(5) ]],
							constant uint const & N [[ buffer(6) ]],
							uint const t [[ thread_position_in_threadgroup ]],
							uint const T [[ threadgroups_per_grid ]]) {
	ushort4 seq = *(constant ushort4*)(seeds+4*t);
	for ( int k = 4 * t, K = N, dk = 4 * T ; k < K ; k += dk ) {
		float4 const x = float4(seq) / 65536.0;
		float4 const n = tanpi(x-0.5);
		float4 const y = fma(n, *(device float4*)(s+k), *(device float4*)(u+k));
		float4 const ju = 1;
		float4 const js = n;
		seq ^= seq << xorshift16.x;
		seq ^= seq >> xorshift16.y;
		seq ^= seq << xorshift16.z;
		switch(min(4, K-k)) {
			case 4:
				*(device float4*)(f+k) = y.xyzw;
				*(device float4*)(gu+k) = ju.xyzw;
				*(device float4*)(gs+k) = js.xyzw;
				break;
			case 3:
				*(device float3*)(f+k) = y.xyz;
				*(device float3*)(gu+k) = ju.xyz;
				*(device float3*)(gs+k) = js.xyz;
				break;
			case 2:
				*(device float2*)(f+k) = y.xy;
				*(device float2*)(gu+k) = ju.xy;
				*(device float2*)(gs+k) = js.xy;
				break;
			case 1:
				*(device float *)(f+k) = y.x;
				*(device float *)(gu+k) = ju.x;
				*(device float *)(gs+k) = js.x;
				break;
		}
	}
}
kernel void CauchyDerivateV(device float * const du [[ buffer(0) ]],
							device float * const ds [[ buffer(1) ]],
							device float const * const f [[ buffer(2) ]],
							device float const * const gu [[ buffer(3) ]],
							device float const * const gs [[ buffer(4) ]],
							device float const * const u [[ buffer(5) ]],
							device float const * const s [[ buffer(6) ]],
							constant uint const & N [[ buffer(7) ]],
							uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const e = du[idx];
		float const v = s[idx];
		du[idx] = e;
		ds[idx] = v - 0.5 * e * e / v;
	}
}
/*----------------------------------------------------------------*/
kernel void CauchyDeltaJV(device float * const d [[ buffer(0) ]],
						  device float const * const ju [[ buffer(1) ]],
						  device float const * const js [[ buffer(2) ]],
						  device float const * const gu [[ buffer(3) ]],
						  device float const * const gs [[ buffer(4) ]],
						  constant uint2 const & S [[ buffer(5) ]],
						  threadgroup float4 * shared [[ threadgroup(0) ]],
						  uint const t [[ thread_position_in_threadgroup ]],
						  uint const T [[ threads_per_threadgroup ]],
						  uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 value = 0;
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = col * size.x + row.x;
		
		value +=
		float4x4(select(0, *(device float4*)(ju+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(ju+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(ju+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(ju+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(gu+k), cols_mask)
		+
		float4x4(select(0, *(device float4*)(js+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(js+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(js+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(js+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(gs+k), cols_mask);
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(d+row.x) += accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(d+row.x) += accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(d+row.x) += accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(d+row.x) += accum->x;
	}
}
kernel void CauchyDeltaJP(device float * const du [[ buffer(0) ]],
						  device float * const ds [[ buffer(1) ]],
						  device float const * const ju [[ buffer(2) ]],
						  device float const * const js [[ buffer(3) ]],
						  device float const * const vu [[ buffer(4) ]],
						  device float const * const vs [[ buffer(5) ]],
						  constant uint2 & S [[ buffer(6) ]],
						  threadgroup float2x4 * shared [[ threadgroup(0) ]],
						  uint const t [[ thread_position_in_threadgroup ]],
						  uint const T [[ threads_per_threadgroup ]],
						  uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float2x4 value = float2x4(0);
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = col * size.x + row.x;
		
		value += float2x4(float4x4(select(0, *(device float4*)(ju + idx.x), rows_mask && cols_mask.x),
								   select(0, *(device float4*)(ju + idx.y), rows_mask && cols_mask.y),
								   select(0, *(device float4*)(ju + idx.z), rows_mask && cols_mask.z),
								   select(0, *(device float4*)(ju + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(vu+k), cols_mask),
						  float4x4(select(0, *(device float4*)(js + idx.x), rows_mask && cols_mask.x),
								   select(0, *(device float4*)(js + idx.y), rows_mask && cols_mask.y),
								   select(0, *(device float4*)(js + idx.z), rows_mask && cols_mask.z),
								   select(0, *(device float4*)(js + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(vs+k), cols_mask));
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float2x4 * accum = shared + a;
	
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(du+row.x) += (*accum)[0].xyzw;
		*(device float4*)(ds+row.x) += (*accum)[1].xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(du+row.x) += (*accum)[0].xyz;
		*(device float3*)(ds+row.x) += (*accum)[1].xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(du+row.x) += (*accum)[0].xy;
		*(device float2*)(ds+row.x) += (*accum)[1].xy;
	} else if ( rows_mask.x ) {
		*(device float *)(du+row.x) += (*accum)[0].x;
		*(device float *)(ds+row.x) += (*accum)[1].x;
	}
}
kernel void CauchyDeltaGP(device float * const du [[ buffer(0) ]],
						  device float * const ds [[ buffer(1) ]],
						  device float const * const ju [[ buffer(2) ]],
						  device float const * const js [[ buffer(3) ]],
						  device float const * const gu [[ buffer(4) ]],
						  device float const * const gs [[ buffer(5) ]],
						  constant uint2 const & N [[ buffer(6) ]],
						  uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		du[idx] += ju[idx] * gu[rows];
		ds[idx] += js[idx] * gs[rows];
	}
}
kernel void CauchyDeltaGV(device float * const d [[ buffer(0) ]],
						  device float const * const ju [[ buffer(1) ]],
						  device float const * const js [[ buffer(2) ]],
						  device float const * const gu [[ buffer(3) ]],
						  device float const * const gs [[ buffer(4) ]],
						  constant uint2 const & N [[ buffer(5) ]],
						  uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		d[idx] += dot(float2(ju[idx], js[idx]), float2(gu[rows], gs[rows]));
	}
}
