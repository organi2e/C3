//
//  CauchyDistributor.metal
//  Distributor
//
//  Created by Kota Nakano on 2017/03/24.
//  Copyright © 2017 Kota Nakano. All rights reserved.
//

// u:     v  =     a  *     x  +     b  *     y  +     c  +     d  *     v
// s: abs(v) = abs(a) * abs(x) + abs(b) * abs(y) + abs(c) + abs(d) * abs(v)

//

#include <metal_stdlib>
using namespace metal;
template<typename T> T const cdf(T const x) {
	return fma(M_1_PI_F, atan(x), 0.5);
}
inline float4x4 const abs(float4x4 const x) {
	return float4x4(abs(x[0]), abs(x[1]), abs(x[2]), abs(x[3]));
}
constant int4 const M_INC = int4(0, 1, 2, 3);
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
	
	if ( !a ) {//avoid over storing, note: alignment of float3 is 16-bytes
		if ( rows_mask.x ) m [ row.x ] += (*accum)[0].x, v[row.x] += (*accum)[1].x;
		if ( rows_mask.y ) m [ row.y ] += (*accum)[0].y, v[row.y] += (*accum)[1].y;
		if ( rows_mask.z ) m [ row.z ] += (*accum)[0].z, v[row.z] += (*accum)[1].z;
		if ( rows_mask.w ) m [ row.w ] += (*accum)[0].w, v[row.w] += (*accum)[1].w;
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
	
	if ( !a ) {
		if ( rows_mask.x ) m[row.x] += (*accum)[0].x, v[row.x] += (*accum)[1].x;
		if ( rows_mask.y ) m[row.y] += (*accum)[0].y, v[row.y] += (*accum)[1].y;
		if ( rows_mask.z ) m[row.z] += (*accum)[0].z, v[row.z] += (*accum)[1].z;
		if ( rows_mask.w ) m[row.w] += (*accum)[0].w, v[row.w] += (*accum)[1].w;
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
//		int const idx = n;
//		s[idx] = sign(s[idx]);
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
	if ( !a ) {
		if ( rows_mask.x ) dx[row.x] += accum->x;
		if ( rows_mask.y ) dx[row.y] += accum->y;
		if ( rows_mask.z ) dx[row.z] += accum->z;
		if ( rows_mask.w ) dx[row.w] += accum->w;
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
		float const p = cdf( u[idx] / s[idx] );
		dx[idx] += ( p - d[idx] ); // p / ( 1 - p );
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
kernel void CauchyConnectX(device float * const ju [[ buffer(0) ]],
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
		float const w = x[cols];
		ju[idx] +=     u[idx];
		js[idx] += abs(s[idx]) * sign(w);
	}
}
kernel void CauchyConnectA(device float * const ju [[ buffer(0) ]],
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
		float const w = x[cols];
		ju[idx] +=     w;
		js[idx] += abs(w) * sign(s[idx]);
	}
}
kernel void CauchyConnectB(device float * const ju [[ buffer(0) ]],
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
		for ( int3 idx = int3(int2(b.x, p.x) * int2(K, N) + int2(p.y, b.y), 0), dx = int3(K, N, 1) ; idx.z < 4 ; idx += dx ) {
			bool4 const bmask = b.x + idx.z < M && p.y + M_INC < K;
			bool4 const pmask = p.x + idx.z < K && b.y + M_INC < N;
			rb[idx.z] = select(0,   (*(device float4*)(Bu+idx.x))*gu, bmask);
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
		float4 const gs = *(device float4*)(Js+p.y)**(device float4*)(Y+p.y);
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
			int const idx = row.y * N + col.y;
			ju [ idx ] += ru [ row.x ] [ col.x ];
			js [ idx ] += rs [ row.x ] [ col.x ];
		}
	}
}
kernel void CauchyConnectC(device float * const ju [[ buffer(0) ]],
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
kernel void CauchyConnectD(device float * const ju [[ buffer(0) ]],
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
kernel void CauchyConnectE(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=     r                  * pu[idx];
		js[idx] += abs(r) * sign(s[rows]) * ps[idx];
	}
}
kernel void CauchyConnectF(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const u [[ buffer(2) ]],
						   device float const * const s [[ buffer(3) ]],
						   constant uint2 const & N [[ buffer(4) ]],
						   uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		js[idx] /= sign(s[rows]);
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
	ushort seq = seeds[t];
	for ( int k = t, K = N ; k < K ; k += T ) {
		float2 const x = float2(u[k], s[k]);
		float2 const j = M_1_PI_F * x / length_squared(x);
		f[k] = step(seq/65536.0, cdf(x.x/x.y));
		//		f[k] = normcdf(x);
		seq ^= seq << xorshift16.x;
		seq ^= seq >> xorshift16.y;
		seq ^= seq << xorshift16.z;
		gu[k] =  j.x;
		gs[k] = -j.y;
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
		float const p = cdf(u[idx]/s[idx]);
		float const g = atan(du[idx]*M_PI_2_F)/M_PI_2_F/p/(1-p);
		du[idx] = g * gu[idx];
		ds[idx] = g * gs[idx];
	}
}
constant float M_1_INT16MAX_F = 1 / 32768.0;
//constant float M_1_UINT16MAX_F = 1 / 65536.0;
kernel void CauchyActivateV(device float * const f [[ buffer(0) ]],
							device float * const gu [[ buffer(1) ]],
							device float * const gs [[ buffer(2) ]],
							device float const * const u [[ buffer(3) ]],
							device float const * const s [[ buffer(4) ]],
							constant ushort const * const seeds [[ buffer(5) ]],
							constant uint const & N [[ buffer(6) ]],
							uint const t [[ thread_position_in_threadgroup ]],
							uint const T [[ threadgroups_per_grid ]]) {
	ushort seq = seeds[t];
	for ( int k = t, K = N ; k < K ; k += T ) {
		float const n = tan(M_PI_2_F*fma(float(seq), M_1_INT16MAX_F, -1));
		float const y = fma(n, s[k], u[k]);
		float const ju = 1;
		float const js = n;
		seq ^= seq << xorshift16.x;
		seq ^= seq >> xorshift16.y;
		seq ^= seq << xorshift16.z;
		f[k] = y;
		gu[k] = ju;
		gs[k] = js;
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
		float const e = du[idx] * gu[idx];
		float const v = s[idx];
		du[idx] = e;
		ds[idx] = v - e * e / v;
	}
}
/*----------------------------------------------------------------*/
kernel void CauchyGradientJV(device float * const d [[ buffer(0) ]],
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
		
		value += float4x4(select(0, *(device float4*)(ju+idx.x), rows_mask && cols_mask.x),
						  select(0, *(device float4*)(ju+idx.y), rows_mask && cols_mask.y),
						  select(0, *(device float4*)(ju+idx.z), rows_mask && cols_mask.z),
						  select(0, *(device float4*)(ju+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(gu+k), cols_mask) +
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
	if ( !a ) {
		if ( rows_mask.x ) d[row.x] += accum->x;
		if ( rows_mask.y ) d[row.y] += accum->y;
		if ( rows_mask.z ) d[row.z] += accum->z;
		if ( rows_mask.w ) d[row.w] += accum->w;
	}
}
kernel void CauchyGradientJP(device float * const du [[ buffer(0) ]],
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
	
	if ( !a ) {
		if ( rows_mask.x ) du[row.x] += (*accum)[0].x, ds[row.x] += (*accum)[1].x;
		if ( rows_mask.y ) du[row.y] += (*accum)[0].y, ds[row.y] += (*accum)[1].y;
		if ( rows_mask.z ) du[row.z] += (*accum)[0].z, ds[row.z] += (*accum)[1].z;
		if ( rows_mask.w ) du[row.w] += (*accum)[0].w, ds[row.w] += (*accum)[1].w;
	}
}
kernel void CauchyGradientGP(device float * const du [[ buffer(0) ]],
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
		du[idx] += d.x;
		ds[idx] += d.y;
	}
}
kernel void CauchyGradientGV(device float * const d [[ buffer(0) ]],
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
