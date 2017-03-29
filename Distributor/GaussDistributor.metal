//
//  GaussDistributor.metal
//  DistributorF
//
//  Created by Kota Nakano on 2017/03/24.
//  Copyright © 2017 Kota Nakano. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;
template<typename T> T erf(T z) {
	T const v = 1.0 / fma(fabs(z), 0.5, 1.0);
	return copysign(fma(-v,
						exp(
							fma(v,
								fma(v,
									fma(v,
										fma(v,
											fma(v,
												fma(v,
													fma(v,
														fma(v,
															fma(v, 0.17087277, -0.82215223),
															1.48851587),
														-1.13520398),
													0.27886807),
												-0.18628806),
											0.09678418),
										0.37409196),
									1.00002368),
								-z*z-1.26551223)),
						1),
					z);
}
template<typename T> inline T sq(const T x) {
	return x * x;
}
inline float4x4 const sq(const float4x4 x) {
	return float4x4(sq(x[0]),
					sq(x[1]),
					sq(x[2]),
					sq(x[3]));
}
constant int4 const M_INC = int4(0, 1, 2, 3);
/*----------------------------------------------------------------*/
kernel void GaussCollectX(device float * const m [[ buffer(0) ]],
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
		value += float2x4(   select(0, *(device float4*)(u+k), cols_mask)  *    x,
						  sq(select(0, *(device float4*)(s+k), cols_mask)) * sq(x));
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
kernel void GaussCollectW(device float * const m [[ buffer(0) ]],
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
		
		value += float2x4(f * float4x4(select(0, *(device float4*)(u+idx.x), rows_mask.x && cols_mask),
									   select(0, *(device float4*)(u+idx.y), rows_mask.y && cols_mask),
									   select(0, *(device float4*)(u+idx.z), rows_mask.z && cols_mask),
									   select(0, *(device float4*)(u+idx.w), rows_mask.w && cols_mask)),
						  sq(f) * float4x4(sq(select(0, *(device float4*)(s+idx.x), rows_mask.x && cols_mask)),
										   sq(select(0, *(device float4*)(s+idx.y), rows_mask.y && cols_mask)),
										   sq(select(0, *(device float4*)(s+idx.z), rows_mask.z && cols_mask)),
										   sq(select(0, *(device float4*)(s+idx.w), rows_mask.w && cols_mask))));
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
kernel void GaussCollectC(device float * const m [[ buffer(0) ]],
						  device float * const v [[ buffer(1) ]],
						  device float const * const u [[ buffer(2) ]],
						  device float const * const s [[ buffer(3) ]],
						  constant uint const & N [[ buffer(4) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		m[idx] +=    u[idx];
		v[idx] += sq(s[idx]);
	}
}
kernel void GaussCollectD(device float * const m [[ buffer(0) ]],
						  device float * const v [[ buffer(1) ]],
						  device float const * const d [[ buffer(2) ]],
						  device float const * const u [[ buffer(3) ]],
						  device float const * const s [[ buffer(4) ]],
						  constant uint const & N [[ buffer(5) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const r = d[idx];
		//		float2 const x =    float2(r, u[idx]);
		//		float2 const y = sq(float2(r, s[idx]));
		//		float2 const z = fma(float2(x.x, y.x), float2(x.y, y.y), float2(m[idx], v[idx]));
		m[idx] +=    r*u[idx];
		v[idx] += sq(r*s[idx]);
	}
}
kernel void GaussCollectF(device float * const u [[ buffer(0) ]],
						  device float * const s [[ buffer(1) ]],
						  constant uint const & N [[ buffer(2) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		s[idx] = sqrt(s[idx]);
	}
}
/*----------------------------------------------------------------*/
kernel void GaussCorrectJ(device float * const dx [[ buffer(0) ]],
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
kernel void GaussCorrectG(device float * const dx [[ buffer(0) ]],
						  device float const * const g [[ buffer(1) ]],
						  constant uint const & N [[ buffer(2) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += g[idx];
	}
}
kernel void GaussCorrectD(device float * const dx [[ buffer(0) ]],
						  device float const  * const f [[ buffer(1) ]],
						  constant float const * const d [[ buffer(2) ]],
						  constant uint const & N [[ buffer(3) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += f[idx] - d[idx];
	}
}
/*----------------------------------------------------------------*/
kernel void GaussJacobianX(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    u[idx];
		js[idx] += sq(s[idx]) * x[cols];
	}
}
kernel void GaussJacobianA(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    v;
		js[idx] += sq(v) * s[idx];
	}
}
kernel void GaussJacobianB(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const Bu [[ buffer(2) ]],
						   device float const * const Bs [[ buffer(3) ]],
						   device float const * const Y [[ buffer(4) ]],
						   device float const * const Ju [[ buffer(5) ]],
						   device float const * const Js [[ buffer(6) ]],
						   device float const * const Pu [[ buffer(7) ]],
						   device float const * const Ps [[ buffer(8) ]],
						   constant uint4 const & mnkl [[ buffer(9) ]],
						   threadgroup float4x4 * const sharedBu [[ threadgroup(0) ]],
						   threadgroup float4x4 * const sharedBs [[ threadgroup(1) ]],
						   threadgroup float4x4 * const sharedPu [[ threadgroup(2) ]],
						   threadgroup float4x4 * const sharedPs [[ threadgroup(3) ]],
						   uint2 const t [[ thread_position_in_threadgroup ]],
						   uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	int const tx = t.x;
	int const ty = t.y;
	threadgroup float4x4 * const cacheBu = sharedBu + tx * L;
	threadgroup float4x4 * const cacheBs = sharedBs + tx * L;
	threadgroup float4x4 * const cachePu = sharedPu + ty * L;
	threadgroup float4x4 * const cachePs = sharedPs + ty * L;
	float4x4 ru(0), rs(0);
	float4x4 rbu, rpu, rbs, rps;
	int2 const b = 4 * int2( g * L + t );
	//	bool4 const brm = b.x + M_INC < M;
	//	bool4 const pcm = b.y + M_INC < N;
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		//		bool4 const prm = p.x + M_INC < K;
		//		bool4 const bcm = p.y + M_INC < K;
		//		for ( int3 row = int3(b.x, p.x, 0) ; row.z < 4 ; ++ row ) {
		//			for ( int3 col = int3(p.y, b.y, 0); col.z < 4 ; ++ col ) {
		//				rbu [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ?    Bu[ row.x * K + col.x ]  * Ju[ col.x ] : 0;
		//				rbs [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ? sq(Bs[ row.x * K + col.x ]) * Js[ col.x ] * Y[ col.x ]: 0;
		//				rpu [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? Pu[ row.y * N + col.y ] : 0;
		//				rps [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? Ps[ row.y * N + col.y ] : 0;
		//			}
		//		}
		float4 const gu = *(device float4*)(Ju+p.y);
		float4 const gs = *(device float4*)(Js+p.y)**(device float4*)(Y+p.y);
		for ( int3 idx = int3(int2(b.x, p.x)*int2(K, N)+int2(p.y, b.y), 0), dx = int3(K, N, 1) ; idx.z < 4 ; idx += dx ) {
			bool4 const bmask = b.x + idx.z < M && p.y + M_INC < K;
			bool4 const pmask = p.x + idx.z < K && b.y + M_INC < N;
			rbu[idx.z] = select(0,   (*(device float4*)(Bu+idx.x))*gu, bmask);
			rbs[idx.z] = select(0, sq(*(device float4*)(Bs+idx.x))*gs, bmask);
			rpu[idx.z] = select(0, *(device float4*)(Pu+idx.y), pmask);
			rps[idx.z] = select(0, *(device float4*)(Ps+idx.y), pmask);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheBu[ty] = rbu;
		cachePu[tx] = rpu;
		cacheBs[ty] = rbs;
		cachePs[tx] = rps;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			rbu = cacheBu[l], rpu = cachePu[l];
			rbs = cacheBs[l], rps = cachePs[l];
			ru[0] += rpu * rbu[0];
			ru[1] += rpu * rbu[1];
			ru[2] += rpu * rbu[2];
			ru[3] += rpu * rbu[3];
			rs[0] += rps * rbs[0];
			rs[1] += rps * rbs[1];
			rs[2] += rps * rbs[2];
			rs[3] += rps * rbs[3];
		}
	}
	for ( int2 row = int2(0, b.x), rows = int2(4, M) ; all(row < rows) ; ++ row ) {
		for ( int2 col = int2(0, b.y), cols = int2(4, N) ; all(col < cols) ; ++ col ) {
			ju [ row.y * N + col.y ] += ru [ row.x ] [ col.x ];
			js [ row.y * N + col.y ] += rs [ row.x ] [ col.x ];
		}
	}
}
kernel void GaussJacobianC(device float * const ju [[ buffer(0) ]],
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
		js[idx] += s[idx];
	}
}
kernel void GaussJacobianD(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const d [[ buffer(2) ]],
						   device float const * const u [[ buffer(3) ]],
						   device float const * const s [[ buffer(4) ]],
						   device float const * const pu [[ buffer(5) ]],
						   device float const * const ps [[ buffer(6) ]],
						   constant uint2 const & N [[ buffer(7) ]],
						   uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		float const r = d[rows];
		ju[idx] +=    r            * pu[idx];
		js[idx] += sq(r) * s[rows] * ps[idx];
	}
}
kernel void GaussJacobianF(device float * const ju [[ buffer(0) ]],
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
kernel void GaussActivateP(device float * const f [[ buffer(0) ]],
						   device float const * const u [[ buffer(1) ]],
						   device float const * const s [[ buffer(2) ]],
						   constant uint const * const seeds [[ buffer(3) ]],
						   constant uint const & N [[ buffer(4) ]],
						   uint const n [[ thread_position_in_grid ]] ) {
	if ( n < N ) {
		int const idx = n;
		float const x = u[idx]/s[idx];
		//f[idx] = step(seeds[idx]/65536.0, fma(erf(M_SQRT1_2_F*x), 65536.0, 65536.0));
		f[idx] = fma(erf(M_SQRT1_2_F*x), 0.5, 0.5);
	}
}
kernel void GaussDerivateP(device float * const du [[ buffer(0) ]],
						   device float * const ds [[ buffer(1) ]],
						   device float * const gu [[ buffer(2) ]],
						   device float * const gs [[ buffer(3) ]],
						   device float const * const u [[ buffer(4) ]],
						   device float const * const s [[ buffer(5) ]],
						   device float const * const d [[ buffer(6) ]],
						   device float const * const b [[ buffer(7) ]],
						   constant uint const & N [[ buffer(8) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const y = s[idx];
		float const x = u[idx] / y;
		float const e = d[idx];//sign(d[idx]);
		float const g = 0.5 * M_2_SQRTPI_F * M_SQRT1_2_F * exp( -0.5 * x * x ) / y;
		du[idx] = e * ( gu[idx] = g );
		ds[idx] = e * ( gs[idx] = g * -x );
	}
}
kernel void GaussDeltaJ(device float * const du [[ buffer(0) ]],
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
		*(device float4*)(du+row.x) = (*accum)[0].xyzw;
		*(device float4*)(ds+row.x) = (*accum)[1].xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(du+row.x) = (*accum)[0].xyz;
		*(device float3*)(ds+row.x) = (*accum)[1].xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(du+row.x) = (*accum)[0].xy;
		*(device float2*)(ds+row.x) = (*accum)[1].xy;
	} else if ( rows_mask.x ) {
		*(device float *)(du+row.x) = (*accum)[0].x;
		*(device float *)(ds+row.x) = (*accum)[1].x;
	}
}
kernel void GaussDeltaGP(device float * const du [[ buffer(0) ]],
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
		float2 const d = float2(ju[idx], js[idx]) * float2(gu[rows], gs[rows]);
		du[idx] = d.x;
		ds[idx] = d.y;
	}
}
kernel void GaussDeltaGV(device float * const d [[ buffer(0) ]],
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
		d[idx] = dot(float2(ju[idx], js[idx]), float2(gu[rows], gs[rows]));
	}
}
