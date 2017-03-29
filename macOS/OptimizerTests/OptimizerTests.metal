//
//  OptimizerTests.metal
//  macOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void dydx(device float * const dydx [[ buffer(0) ]],
				 device float const * const x [[ buffer(1) ]],
				 uint i [[ thread_position_in_grid ]],
				 uint I [[ threads_per_grid ]]) {
	const float w[8] = {
		100000,
		1000,
		10,
		1,
		1,
		0.1,
		0.001,
		0.00001
	};
	/*
	 const float w[8] = {
		1,
		1,
		1,
		1,
		1,
		1,
		1,
		1
	 };*/
	dydx[i] = w[7-i%8] * (0.7+0.3*cos(x[(i+13)%I]*x[(i+62)%I]/1080.0)) * ( x[i] - i - 1 + 64 );
}
kernel void dydx2(device float * const dydx [[ buffer(0) ]],
				  device float const * const x [[ buffer(1) ]],
				  uint i [[ thread_position_in_grid ]],
				  uint I [[ threads_per_grid ]]) {
	const float w[8] = {
		10000,
		1000,
		10,
		1,
		1,
		0.1,
		0.001,
		0.0001
	};
	/*
	 const float w[8] = {
		1,
		1,
		1,
		1,
		1,
		1,
		1,
		1
	 };*/
	dydx[i] = w[7-(i+3)%8] * (0.55+0.45*cos(x[(i+18)%I]*x[(i+180)%I]/512.0)) * ( x[i] - i - 1 + 64 );
}
