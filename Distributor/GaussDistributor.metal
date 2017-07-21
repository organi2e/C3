//
//  GaussDistributor.metal
//  Distributor
//
//  Created by Kota Nakano on 2017/03/24.
//  Copyright © 2017 Kota Nakano. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;
template<typename T> T erf(T const z) {
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
																													fma(v, 0.17087277,
																															-0.82215223),
																													1.48851587),
																											-1.13520398),
																									0.27886807),
																							-0.18628806),
																					0.09678418),
																			0.37409196),
																	1.00002368),
															-fma(z, z, 1.26551223))),
											1),
									z);
}
//refer: Mike Giles, ``Approximating the erfinv function, ''
template<typename T> T erfinv(T const x) {
	T const z = - log ( 1 - x * x );
	auto const s = z < 5.0;
	T const w = select(sqrt(z) - 3.0, z - 2.5, s);
	T const y = fma(w,
									fma(w,
											fma(w,
													fma(w,
															fma(w,
																	fma(w,
																			fma(w,
																					fma(w, select(T(-0.000200214257), T(2.81022636e-08), s),
																							select(T(0.000100950558), T(3.43273939e-07), s)),
																					select(T(0.00134934322), T(-3.5233877e-06), s)),
																			select(T(-0.00367342844), T(-4.39150654e-06), s)),
																	select(T(0.00573950773), T(0.00021858087e-00), s)),
															select(T(-0.0076224613), T(-0.00125372503e-00), s)),
													select(T(0.00943887047), T(-0.00417768164e-00), s)),
											select(T(1.00167406), T(0.246640727e-00), s)),
									select(T(2.83297682), T(1.50140941e-00), s));
	return x * y;
}
/*
 template<typename T> T erfinv(T const x) {
	T const z = - log ( 1 - x * x );
	if ( z < 5.0 ) {
 T const w = z - 2.5;
 return x * fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w, 2.81022636e-08,
 3.43273939e-07),
 -3.5233877e-06),
 -4.39150654e-06),
 0.00021858087),
 -0.00125372503),
 -0.00417768164),
 0.246640727),
 1.50140941);
	}
	else {
 T const w = sqrt(z) - 3.0;
 return x * fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w,
 fma(w, -0.000200214257,
 0.000100950558),
 0.00134934322),
 -0.00367342844),
 0.00573950773),
 -0.0076224613),
 0.00943887047),
 1.00167406),
 2.83297682);
	}
 }
 */
template<typename T> T const cdf(T const x) {
	return fma(erf(M_SQRT1_2_F*x), 32767.984375/65536.0, 0.5);
}

template<typename T> T const boxmuller(T const x) { return sqrt(-2*log(-x.xy)).xxyy * float2(cospi(2*x.z), sinpi(2*x.w)).xyxy; }
template<>half4 const boxmuller(half4 const);
template<>float4 const boxmuller(float4 const);

template<typename T> inline T sq(const T x) { return x * x; }

inline half2x2 const sq(half2x2 const x) { return half2x2(sq(x[0]), sq(x[1])); }
inline half3x3 const sq(half3x3 const x) { return half3x3(sq(x[0]), sq(x[1]), sq(x[2])); }
inline half4x4 const sq(half4x4 const x) { return half4x4(sq(x[0]), sq(x[1]), sq(x[2]), sq(x[3])); }

inline float2x2 const sq(float2x2 const x) { return float2x2(sq(x[0]), sq(x[1])); }
inline float3x3 const sq(float3x3 const x) { return float3x3(sq(x[0]), sq(x[1]), sq(x[2])); }
inline float4x4 const sq(float4x4 const x) { return float4x4(sq(x[0]), sq(x[1]), sq(x[2]), sq(x[3])); }

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
	
	if ( !a ) {//avoid over storing, note: alignment of float3 is 16-bytes same as float4
		if ( rows_mask.x ) m [ row.x ] += (*accum)[0].x, v[row.x] += (*accum)[1].x;
		if ( rows_mask.y ) m [ row.y ] += (*accum)[0].y, v[row.y] += (*accum)[1].y;
		if ( rows_mask.z ) m [ row.z ] += (*accum)[0].z, v[row.z] += (*accum)[1].z;
		if ( rows_mask.w ) m [ row.w ] += (*accum)[0].w, v[row.w] += (*accum)[1].w;
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
		
		value += float2x4(   f *     float4x4(select(0, *(device float4*)(u+idx.x), rows_mask.x && cols_mask),
																					select(0, *(device float4*)(u+idx.y), rows_mask.y && cols_mask),
																					select(0, *(device float4*)(u+idx.z), rows_mask.z && cols_mask),
																					select(0, *(device float4*)(u+idx.w), rows_mask.w && cols_mask)),
											sq(f) * sq(float4x4(select(0, *(device float4*)(s+idx.x), rows_mask.x && cols_mask),
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
	if ( !a ) {
		if ( rows_mask.x ) dx[row.x] += accum->x;
		if ( rows_mask.y ) dx[row.y] += accum->y;
		if ( rows_mask.z ) dx[row.z] += accum->z;
		if ( rows_mask.w ) dx[row.w] += accum->w;
	}
}
kernel void GaussCorrectG(device float * const dx [[ buffer(0) ]],
													device float const * const x [[ buffer(1) ]],
													device float const * const d [[ buffer(2) ]],
													constant uint const & N [[ buffer(3) ]],
													uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += ( x[idx] - d[idx] );
	}
}
kernel void GaussCorrectN(device float * const dx [[ buffer(0) ]],
													device float const * const x [[ buffer(1) ]],
													constant uint const & N [[ buffer(2) ]],
													uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += 2 * x[idx] - 1;
	}
}
kernel void GaussCorrectP(device float * const dx [[ buffer(0) ]],
													device float const * const u [[ buffer(1) ]],
													device float const * const s [[ buffer(2) ]],
													device float const * const d [[ buffer(3) ]],
													constant uint const & N [[ buffer(4) ]],
													uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += cdf( u[idx] / s[idx] ) - d[idx];
	}
}
kernel void GaussCorrectV(device float * const dx [[ buffer(0) ]],
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
kernel void GaussConnectX(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    u[idx];
		js[idx] += sq(s[idx]) * w;
	}
}
kernel void GaussConnectA(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    w;
		js[idx] += sq(w) * s[idx];
	}
}
kernel void GaussConnectB(device float * const ju [[ buffer(0) ]],
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
			rb[idx.z] = select(0, sq(*(device float4*)(Bs+idx.x))*gs, bmask);
			rp[idx.z] = select(0, *(device float4*)(Ps+idx.y), pmask);
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
kernel void GaussConnectC(device float * const ju [[ buffer(0) ]],
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
		js[idx] += s[rows];
	}
}
kernel void GaussConnectD(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    u[rows];
		js[idx] += sq(s[rows]) * d[rows];
	}
}
kernel void GaussConnectE(device float * const ju [[ buffer(0) ]],
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
		ju[idx] +=    r            * pu[idx];
		js[idx] += sq(r) * s[rows] * ps[idx];
	}
}
kernel void GaussConnectF(device float * const ju [[ buffer(0) ]],
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
constant float M_SQRT1_2PI_F = 0.5 * M_2_SQRTPI_F * M_SQRT1_2_F;
kernel void GaussActivateP(device float * const f [[ buffer(0) ]],
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
		float const r = 1 / s[k];
		float const x = r * u[k];
		float const ju = M_SQRT1_2PI_F * exp ( -0.5 * x * x ) * r;
		float const js = ju * -x;
		f[k] = step(float(seq), cdf(x) * 65536.0);
//		f[k] = cdf(x);
		seq ^= seq << xorshift16.x;
		seq ^= seq >> xorshift16.y;
		seq ^= seq << xorshift16.z;
		gu[k] = ju;
		gs[k] = js;
	}
}
kernel void GaussDerivateP(device float * const du [[ buffer(0) ]],
													 device float * const ds [[ buffer(1) ]],
													 device float const * const d [[ buffer(2) ]],
													 device float const * const f [[ buffer(3) ]],
													 device float const * const gu [[ buffer(4) ]],
													 device float const * const gs [[ buffer(5) ]],
													 device float const * const u [[ buffer(6) ]],
													 device float const * const s [[ buffer(7) ]],
													 constant uint const & N [[ buffer(8) ]],
													 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const p = cdf( u[idx] / s[idx] );
//		float const g = d[idx] / p / ( 1 - p );
		float const g = erf( d[idx] / M_2_SQRTPI_F ) / p / ( 1 - p );
		du[idx] = g * gu[idx];
		ds[idx] = g * gs[idx];
	}
}
constant float M_1_INT16MAX_F = 1 / 32768.0;
kernel void GaussActivateV(device float * const f [[ buffer(0) ]],
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
		float const n = erfinv(fma(float(seq), M_1_INT16MAX_F, -1)) * M_SQRT2_F;
		float const y = fma(n, s[k], u[k]);
		seq ^= seq << xorshift16.x;
		seq ^= seq >> xorshift16.y;
		seq ^= seq << xorshift16.z;
		f[k] = y;
		gu[k] = 1;
		gs[k] = n;
	}
}
kernel void GaussDerivateV(device float * const du [[ buffer(0) ]],
													 device float * const ds [[ buffer(1) ]],
													 device float const * const d [[ buffer(2) ]],
													 device float const * const f [[ buffer(3) ]],
													 device float const * const gu [[ buffer(4) ]],
													 device float const * const gs [[ buffer(5) ]],
													 device float const * const u [[ buffer(6) ]],
													 device float const * const s [[ buffer(7) ]],
													 constant uint const & N [[ buffer(8) ]],
													 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const e = d[idx], ee = e * e;
		float const v = s[idx], vv = v * v;
		float const dKLdU = e / vv;
		float const dKLdS = ( vv - ee ) / vv / v;
		du[idx] = select(0.0, dKLdU, isfinite(dKLdU));
		ds[idx] = select(0.0, dKLdS, isfinite(dKLdS));
	}
}
kernel void GaussActivateX(device float * const p [[ buffer(0) ]],
													 device float const * const u [[ buffer(1) ]],
													 device float const * const s [[ buffer(2) ]],
													 constant uint const & N [[ buffer(3) ]],
													 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		p[idx] = cdf(u[idx]/s[idx]);
	}
}
kernel void GaussDerivateN(device float * const du [[ buffer(0) ]],
													 device float * const ds [[ buffer(1) ]],
													 device float2 * const momentum [[ buffer(2) ]],
													 device float4 * const gradient [[ buffer(3) ]],
													 device float const * const u [[ buffer(4) ]],
													 device float const * const s [[ buffer(5) ]],
													 constant float const & gamma [[ buffer(6) ]],
													 constant uint const & N [[ buffer(7) ]],
													 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		
		int const idx = n;
		
		float const r = gamma;
		
		float2 const x = float2(u[idx], s[idx]);
		
		float2 const m = mix(momentum[idx], float2(x.x, length_squared(x)), r);
		float4 const g = mix(gradient[idx], float4(1, 0, 2 * x.x, 2 * x.y), r);
		
		float const v = m.y - m.x * m.x, vv = v * v;
		float const uu = m.x * m.x + 1;
		
		float const dKLdU = g.z * ( v - uu ) / vv + 2 * g.x * m.x * uu / vv;
		float const dKLdS = g.w * ( v - uu ) / vv;
		
//		float const dKLdU = g.x * ( m.y - 2 * m.x * m.x - 1 ) / v / v + g.z * 2 * m.x * ( m.x * m.x + 1 ) / v / v;
//		float const dKLdS = g.w * ( m.y - 2 * m.x * m.x - 1 ) / v / v;
		
		du[idx] += select(0.0, dKLdU, isfinite(dKLdU));
		ds[idx] += select(0.0, dKLdS, isfinite(dKLdS));
		
		momentum[idx] = m;
		gradient[idx] = g;
		
	}
}
/*----------------------------------------------------------------*/
kernel void GaussGradientJV(device float * const d [[ buffer(0) ]],
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
kernel void GaussGradientJP(device float * const du [[ buffer(0) ]],
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
kernel void GaussGradientGP(device float * const du [[ buffer(0) ]],
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
kernel void GaussGradientGV(device float * const d [[ buffer(0) ]],
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
