//
//  Degenerate.metal
//  macOS
//
//  Created by Kota Nakano on 2017/04/10.
//
//

#include <metal_stdlib>
using namespace metal;
constant int4 const M_INC = int4(0, 1, 2, 3);
/*----------------------------------------------------------------*/
kernel void DegenerateCollectW(device float * const v [[ buffer(0) ]],
							   device float const * const w [[ buffer(1) ]],
							   device float const * const x [[ buffer(2) ]],
							   constant uint2 & S [[ buffer(3) ]],
							   threadgroup float4 * shared [[ threadgroup(0) ]],
							   uint const t [[ thread_position_in_threadgroup ]],
							   uint const T [[ threads_per_threadgroup ]],
							   uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 value = float4(0);
	
	int4 const row = 4 * n + M_INC;
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + M_INC;
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = row * size.y + k;
		value +=
		select(0, *(device float4*)(x + k), cols_mask) * float4x4(select(0, *(device float4*)(w+idx.x), rows_mask.x && cols_mask),
																  select(0, *(device float4*)(w+idx.y), rows_mask.y && cols_mask),
																  select(0, *(device float4*)(w+idx.z), rows_mask.z && cols_mask),
																  select(0, *(device float4*)(w+idx.w), rows_mask.w && cols_mask));
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
		*(device float4*)(v+row.x) += accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(v+row.x) += accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(v+row.x) += accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(v+row.x) += accum->x;
	}
}
kernel void DegenerateCollectC(device float * const v [[ buffer(0) ]],
							   device float const * const c [[ buffer(1) ]],
							   constant uint const & N [[ buffer(2) ]],
							   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		v[idx] += c[idx];
	}
}
kernel void DegenerateCollectD(device float * const v [[ buffer(0) ]],
							   device float const * const d [[ buffer(1) ]],
							   device float const * const p [[ buffer(2) ]],
							   constant uint const & N [[ buffer(3) ]],
							   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		v[idx] += d[idx] * p[idx];
	}
}
kernel void DegenerateCollectF(device float * const u [[ buffer(0) ]],
							   constant uint const & N [[ buffer(1) ]],
							   uint const n [[ thread_position_in_grid ]]) {
	
}
/*----------------------------------------------------------------*/
kernel void DegenerateCorrectJ(device float * const dx [[ buffer(0) ]],
							   device float const * const j [[ buffer(1) ]],
							   device float const * const g [[ buffer(2) ]],
							   constant uint2 const & S [[ buffer(3) ]],
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
		float4x4(select(0, *(device float4*)(j+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(j+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(j+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(j+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(g+k), cols_mask);
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
kernel void DegenerateCorrectG(device float * const dx [[ buffer(0) ]],
							   device float const * const x [[ buffer(1) ]],
							   device float const * const d [[ buffer(2) ]],
							   constant uint const & N [[ buffer(3) ]],
							   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += (x[idx] - d[idx]);
	}
}
kernel void DegenerateCorrectN(device float * const dx [[ buffer(0) ]],
							   device float const * const x [[ buffer(1) ]],
							   constant uint const & N [[ buffer(2) ]],
							   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		dx[idx] += 2 * x[idx] - 1;
	}
}
/*----------------------------------------------------------------*/
kernel void DegenerateJacobianX(device float * const j [[ buffer(0) ]],
								device float const * const x [[ buffer(1) ]],
								device float const * const w [[ buffer(2) ]],
								constant uint2 const & N [[ buffer(3) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		j[idx] += w[idx];
	}
}
kernel void DegenerateJacobianA(device float * const j [[ buffer(0) ]],
								device float const * const w [[ buffer(1) ]],
								device float const * const x [[ buffer(2) ]],
								constant uint2 const & N [[ buffer(3) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		j[idx] += x[cols];
	}
}
kernel void DegenerateJacobianB(device float * const j [[ buffer(0) ]],
								device float const * const B [[ buffer(1) ]],
								device float const * const Y [[ buffer(2) ]],
								device float const * const J [[ buffer(3) ]],
								device float const * const P [[ buffer(4) ]],
								constant uint4 const & mnkl [[ buffer(5) ]],
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
	thread float4x4 r(0), rb, rp;
	int2 const b = 4 * int2( g * L + t );
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		float4 const g = *(device float4*)(J+p.y);
		for ( int3 idx = int3(int2(b.x, p.x) * int2(K, N) + int2(p.y, b.y), 0), dx = int3(K, N, 1) ; idx.z < 4 ; idx += dx ) {
			bool4 const bmask = b.x + idx.z < M && p.y + M_INC < K;
			bool4 const pmask = p.x + idx.z < K && b.y + M_INC < N;
			rb[idx.z] = select(0, (*(device float4*)(B+idx.x))*g, bmask);
			rp[idx.z] = select(0, *(device float4*)(P+idx.y), pmask);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rb;
		cacheP[tx] = rp;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			rb = cacheB[l], rp = cacheP[l];
			r[0] += rp * rb[0];
			r[1] += rp * rb[1];
			r[2] += rp * rb[2];
			r[3] += rp * rb[3];
		}
	}
	for ( int2 row = int2(0, b.x), rows = int2(4, M) ; all(row < rows) ; ++ row ) {
		for ( int2 col = int2(0, b.y), cols = int2(4, N) ; all(col < cols) ; ++ col ) {
			j [ row.y * N + col.y ] += r [ row.x ] [ col.x ];
		}
	}
}
kernel void DegenerateJacobianC(device float * const j [[ buffer(0) ]],
								device float const * const c [[ buffer(1) ]],
								constant uint2 const & N [[ buffer(2) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		j[idx] += 1;
	}
}
kernel void DegenerateJacobianD(device float * const j [[ buffer(0) ]],
								device float const * const d [[ buffer(1) ]],
								device float const * const v [[ buffer(2) ]],
								constant uint2 const & N [[ buffer(3) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		j[idx] += v[rows];
	}
}
kernel void DegenerateJacobianE(device float * const j [[ buffer(0) ]],
								device float const * const v [[ buffer(1) ]],
								device float const * const d [[ buffer(2) ]],
								device float const * const p [[ buffer(3) ]],
								constant uint2 const & N [[ buffer(4) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		j[idx] += d[rows] * p[idx];
	}
}
kernel void DegenerateJacobianF(device float * const j [[ buffer(0) ]],
								device float const * const v [[ buffer(1) ]],
								constant uint2 const & N [[ buffer(2) ]],
								uint2 const n [[ thread_position_in_grid ]]) {
	
}
/*----------------------------------------------------------------*/
kernel void DegenerateActivateP(device float * const f [[ buffer(0) ]],
								device float * const g [[ buffer(1) ]],
								device float const * const v [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const x = v[idx];
		float const p = 1 / ( 1 + exp(-x) );
		f[idx] = step(0, x);
		g[idx] = p * ( 1 - p );
	}
}
kernel void DegenerateDerivateP(device float * const d [[ buffer(0) ]],
								device float const * const g [[ buffer(1) ]],
								device float const * const v [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		d[idx] = sign(d[idx]);
	}
}
kernel void DegenerateActivateV(device float * const f [[ buffer(0) ]],
								device float * const g [[ buffer(1) ]],
								device float const * const v [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		f[idx] = v[idx];
		g[idx] = 1;
	}
}
kernel void DegenerateDerivateV(device float * const d [[ buffer(0) ]],
								device float const * const g [[ buffer(1) ]],
								device float const * const u [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		
	}
}
/*----------------------------------------------------------------*/
kernel void DegenerateDeltaJ(device float * const d [[ buffer(0) ]],
							 device float const * const j [[ buffer(1) ]],
							 device float const * const g [[ buffer(2) ]],
							 constant uint2 const & S [[ buffer(3) ]],
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
		float4x4(select(0, *(device float4*)(j+idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(j+idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(j+idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(j+idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(g+k), cols_mask);
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
kernel void DegenerateDeltaG(device float * const d [[ buffer(0) ]],
							 device float const * const j [[ buffer(1) ]],
							 device float const * const g [[ buffer(2) ]],
							 constant uint2 const & N [[ buffer(3) ]],
							 uint2 const n [[ thread_position_in_grid ]]) {
	if ( n.x < N.x ) {
		int const rows = n.x;
		int const cols = n.y;
		int const idx = rows * N.y + cols;
		d[idx] = j[idx] * g[rows];
	}
}
