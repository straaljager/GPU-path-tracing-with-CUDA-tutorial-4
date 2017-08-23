// BVH traversal kernels based on "Understanding the 

#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "CudaRenderKernel.h"
#include "stdio.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cutil_math.h"  // required for float3

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays
#define samps 1
#define F32_MIN          (1.175494351e-38f)
#define F32_MAX          (3.402823466e+38f)

#define HDRwidth 3200
#define HDRheight 1600
#define HDR
#define EntrypointSentinel 0x76543210
#define MaxBlockHeight 6

enum Refl_t { DIFF, METAL, SPEC, REFR, COAT };  // material types

// CUDA textures containing scene data
texture<float4, 1, cudaReadModeElementType> bvhNodesTexture;
texture<float4, 1, cudaReadModeElementType> triWoopTexture;
texture<float4, 1, cudaReadModeElementType> triNormalsTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;
texture<float4, 1, cudaReadModeElementType> HDRtexture;

__device__ inline Vec3f absmax3f(const Vec3f& v1, const Vec3f& v2){
	return Vec3f(v1.x*v1.x > v2.x*v2.x ? v1.x : v2.x, v1.y*v1.y > v2.y*v2.y ? v1.y : v2.y, v1.z*v1.z > v2.z*v2.z ? v1.z : v2.z);
}

struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

		// ray/sphere intersection
		float3 op = pos - r.orig;   
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant of quadratic formula
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0.0f);
	}
};

__constant__ Sphere spheres[] = {
	// sun
	//{ 10000, { 50.0f, 40.8f, -1060 }, { 0.3, 0.3, 0.3 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // sky   0.003, 0.003, 0.003	
	//{ 4.5, { 0.0f, 12.5, 0 }, { 6, 4, 1 }, { .6f, .6f, 0.6f }, DIFF },  /// lightsource	
	{ 10000.02, { 50.0f, -10001.35, 0 }, { 0.0, 0.0, 0 }, { 0.3f, 0.3f, 0.3f }, DIFF }, // ground  300/-301.0
	//{ 10000, { 50.0f, -10000.1, 0 }, { 0, 0, 0 }, { 0.3f, 0.3f, 0.3f }, DIFF }, // double shell to prevent light leaking
	//{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },  // horizon brightener
	
	//{ 0.5, { 30.0f, 180.5, 42 }, { 0, 0, 0 }, { .6f, .6f, 0.6f }, DIFF },  // small sphere 1  
	//{ 0.8, { 2.0f, 0.f, 0 }, { 0.0, 0.0, 0.0 }, { 0.8f, 0.8f, 0.8f }, SPEC },  // small sphere 2
	//{ 0.8, { -3.0f, 0.f, 0 }, { 0.0, 0.0, 0.0 }, { 0.0f, 0.0f, 0.2f }, COAT },  // small sphere 2
	{ 2.5, { -6.0f, 0.5f, 0.0f }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f }, SPEC },  // small sphere 2
	//{ 0.6, { -10.0f, -2.f, 1.0f }, { 0.0, 0.0, 0.0 }, { 0.8f, 0.8f, 0.8f }, DIFF },  // small sphere 2
	//{ 0.8, { -1.0f, -0.7f, 4.0f }, { 0.0, 0.0, 0.0 }, { 0.8f, 0.8f, 0.8f }, REFR },  // small sphere 2
	//{ 9.4, { 9.0f, 0.f, -9.0f }, { 0.0, 0.0, 0.0 }, { 0.8f, 0.8f, 0.f }, DIFF },  // small sphere 2
	//{ 22, { 105.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, DIFF }, // small sphere 3
};


//  RAY BOX INTERSECTION ROUTINES

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.

// float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
// float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)

// Perform min/max operations in hardware
// Using Kepler's video instructions, see http://docs.nvidia.com/cuda/parallel-thread-execution/#axzz3jbhbcTZf																			//  : "=r"(v) overwrites v and puts it in a register
// see https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

// standard ray box intersection routines (for debugging purposes only)
// based on Intersect::RayBox() in original Aila/Laine code
__device__ __inline__ float spanBeginKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d){ 

	Vec3f t0 = Vec3f(lo_x, lo_y, lo_z);
	Vec3f t1 = Vec3f(hi_x, hi_y, hi_z);
	
	Vec3f realmin = min3f(t0, t1);

	float raybox_tmin = realmin.max(); // maxmin

	//return Vec2f(tmin, tmax);
	return raybox_tmin;
}

__device__ __inline__ float spanEndKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d){

	Vec3f t0 = Vec3f(lo_x, lo_y, lo_z);
	Vec3f t1 = Vec3f(hi_x, hi_y, hi_z);

	Vec3f realmax = max3f(t0, t1);

	float raybox_tmax = realmax.min(); /// minmax

	//return Vec2f(tmin, tmax);
	return raybox_tmax;
}

__device__ __inline__ void swap2(int& a, int& b){ int temp = a; a = b; b = temp;}

// standard ray triangle intersection routines (for debugging purposes only)
// based on Intersect::RayTriangle() in original Aila/Laine code
__device__ Vec3f intersectRayTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec4f& rayorig, const Vec4f& raydir){
	
	const Vec3f rayorig3f = Vec3f(rayorig.x, rayorig.y, rayorig.z);
	const Vec3f raydir3f = Vec3f(raydir.x, raydir.y, raydir.z);

	const float EPSILON = 0.00001f; // works better
	const Vec3f miss(F32_MAX, F32_MAX, F32_MAX);
	
	float raytmin = rayorig.w;
	float raytmax = raydir.w;

	Vec3f edge1 = v1 - v0;
	Vec3f edge2 = v2 - v0;
	
	Vec3f tvec = rayorig3f - v0;
	Vec3f pvec = cross(raydir3f, edge2);
	float det = dot(edge1, pvec);
	
	float invdet = 1.0f / det;
	
	float u = dot(tvec, pvec) * invdet;
	
	Vec3f qvec = cross(tvec, edge1);
	
	float v = dot(raydir3f, qvec) * invdet;

	if (det > EPSILON)
	{
		if (u < 0.0f || u > 1.0f) return miss; // 1.0 want = det * 1/det  
		if (v < 0.0f || (u + v) > 1.0f) return miss;
		// if u and v are within these bounds, continue and go to float t = dot(...	           
	}

	else if (det < -EPSILON)
	{
		if (u > 0.0f || u < 1.0f) return miss;
		if (v > 0.0f || (u + v) < 1.0f) return miss;
		// else continue
	}

	else // if det is not larger (more positive) than EPSILON or not smaller (more negative) than -EPSILON, there is a "miss"
		return miss;

	float t = dot(edge2, qvec) * invdet;

	if (t > raytmin && t < raytmax)
		return Vec3f(u, v, t);
	
	// otherwise (t < raytmin or t > raytmax) miss
	return miss;
}

// modified intersection routine (uses regular instead of woopified triangles) for debugging purposes

__device__ void DEBUGintersectBVHandTriangles(const float4 rayorig, const float4 raydir,
	const float4* gpuNodes, const float4* gpuTriWoops, const float4* gpuDebugTris, const int* gpuTriIndices,
	int& hitTriIdx, float& hitdistance, int& debugbingo, Vec3f& trinormal, int leafcount, int tricount, bool needClosestHit){

	int traversalStack[STACK_SIZE];

	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / dir
	float   oodx, oody, oodz;       // orig / dir

	char*   stackPtr;
	int		leafAddr;
	int		nodeAddr;
	int     hitIndex;
	float	hitT;
	int threadId1;
	
	threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));

	origx = rayorig.x;
	origy = rayorig.y;
	origz = rayorig.z;
	dirx = raydir.x;
	diry = raydir.y;
	dirz = raydir.z;
	tmin = rayorig.w;

	// ooeps is very small number, used instead of raydir xyz component when that component is near zero
	float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
	idirx = 1.0f / (fabsf(raydir.x) > ooeps ? raydir.x : copysignf(ooeps, raydir.x)); // inverse ray direction
	idiry = 1.0f / (fabsf(raydir.y) > ooeps ? raydir.y : copysignf(ooeps, raydir.y)); // inverse ray direction
	idirz = 1.0f / (fabsf(raydir.z) > ooeps ? raydir.z : copysignf(ooeps, raydir.z)); // inverse ray direction
	oodx = origx * idirx;  // ray origin / ray direction
	oody = origy * idiry;  // ray origin / ray direction
	oodz = origz * idirz;  // ray origin / ray direction

	traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 is 1985229328 in decimal
	stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
	leafAddr = 0;   // No postponed leaf.
	nodeAddr = 0;   // Start from the root.
	hitIndex = -1;  // No triangle intersected so far.
	hitT = raydir.w;

	while (nodeAddr != EntrypointSentinel) // EntrypointSentinel = 0x76543210 
	{
		// Traverse internal nodes until all SIMD lanes have found a leaf.

		bool searchingLeaf = true; // flag required to increase efficiency of threads in warp
		while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)   
		{
			float4* ptr = (float4*)((char*)gpuNodes + nodeAddr);				
			float4 n0xy = ptr[0]; // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)		
			float4 n1xy = ptr[1]; // childnode 1. xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)		
			float4 nz = ptr[2]; // childnodes 0 and 1, z-bounds(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)			

			// ptr[3] contains indices to 2 childnodes in case of innernode, see below
			// (childindex = size of array during building, see CudaBVH.cpp)

			// compute ray intersections with BVH node bounding box

			float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
			float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
			float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
			float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
			float c0loz = nz.x   * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
			float c0hiz = nz.y   * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
			float c1loz = nz.z   * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
			float c1hiz = nz.w   * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
			float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
			float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
			float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
			float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
			float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
			float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
			float c1min = spanBeginKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

			float ray_tmax = 1e20;
			bool traverseChild0 = (c0min <= c0max) && (c0min >= tmin) && (c0min <= ray_tmax);
			bool traverseChild1 = (c1min <= c1max) && (c1min >= tmin) && (c1min <= ray_tmax);

			if (!traverseChild0 && !traverseChild1)  
			{
				nodeAddr = *(int*)stackPtr; // fetch next node by popping stack
				stackPtr -= 4; // popping decrements stack by 4 bytes (because stackPtr is a pointer to char) 
			}

			// Otherwise => fetch child pointers.

			else  // one or both children intersected
			{
				int2 cnodes = *(int2*)&ptr[3];
				// set nodeAddr equal to intersected childnode (first childnode when both children are intersected)
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;  

				// Both children were intersected => push the farther one on the stack.

				if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
				{   
					if (c1min < c0min)  
						swap2(nodeAddr, cnodes.y); 
					stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
					*(int*)stackPtr = cnodes.y; // push furthest node on the stack
				}
			}

			// First leaf => postpone and continue traversal.
			// leafnodes have a negative index to distinguish them from inner nodes
			// if nodeAddr less than 0 -> nodeAddr is a leaf
			if (nodeAddr < 0 && leafAddr >= 0)  // if leafAddr >= 0 -> no leaf found yet (first leaf)
			{
				searchingLeaf = false; // required for warp efficiency
				leafAddr = nodeAddr;  
	
				nodeAddr = *(int*)stackPtr;  // pops next node from stack
				stackPtr -= 4;  // decrement by 4 bytes (stackPtr is a pointer to char)
			}

			// All SIMD lanes have found a leaf => process them.
			// NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
			// tried everything with CUDA 4.2 but always got several redundant instructions.

			// if (!searchingLeaf){ break;  }  

			// if (!__any(searchingLeaf)) break; // "__any" keyword: if none of the threads is searching a leaf, in other words
			// if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

			// if(!__any(leafAddr >= 0))   /// als leafAddr in PTX code >= 0, dan is het geen echt leafNode   
			//    break;

			unsigned int mask; // mask replaces searchingLeaf in PTX code

			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
			"}"
				: "=r"(mask)
				: "r"(leafAddr));

			if (!mask)
				break;
		}

		///////////////////////////////////////
		/// LEAF NODE / TRIANGLE INTERSECTION
		///////////////////////////////////////

		while (leafAddr < 0)  // if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode
		{    
			// leafAddr is stored as negative number, see cidx[i] = ~triWoopData.getSize(); in CudaBVH.cpp
		
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{    // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

				// Read first 16 bytes of the triangle.
				// fetch first triangle vertex
				float4 v0f = gpuDebugTris[triAddr + 0];
	
				// End marker 0x80000000 (= negative zero) => all triangles in leaf processed. --> terminate 				
				if (__float_as_int(v0f.x) == 0x80000000) break; 
					
				float4 v1f = gpuDebugTris[triAddr + 1];
				float4 v2f = gpuDebugTris[triAddr + 2];

				const Vec3f v0 = Vec3f(v0f.x, v0f.y, v0f.z);
				const Vec3f v1 = Vec3f(v1f.x, v1f.y, v1f.z);
				const Vec3f v2 = Vec3f(v2f.x, v2f.y, v2f.z);

				// convert float4 to Vec4f

				Vec4f rayorigvec4f = Vec4f(rayorig.x, rayorig.y, rayorig.z, rayorig.w);
				Vec4f raydirvec4f = Vec4f(raydir.x, raydir.y, raydir.z, raydir.w);

				Vec3f bary = intersectRayTriangle(v0, v1, v2, rayorigvec4f, raydirvec4f);

				float t = bary.z; // hit distance along ray

				if (t > tmin && t < hitT)   // if there is a miss, t will be larger than hitT (ray.tmax)
					{								
						hitIndex = triAddr;
						hitT = t;  /// keeps track of closest hitpoint

						trinormal = cross(v0 - v1, v0 - v2);
						
						if (!needClosestHit){  // shadow rays only require "any" hit with scene geometry, not the closest one
							nodeAddr = EntrypointSentinel;
							break;
						}
					}

				} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;

			if (nodeAddr < 0)
			{
				nodeAddr = *(int*)stackPtr;  // pop stack
				stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
			}
		} // end leaf/triangle intersection loop
	} // end of node traversal loop

	// Remap intersected triangle index, and store the result.

	if (hitIndex != -1){  
		// remapping tri indices delayed until this point for performance reasons
		// (slow global memory lookup in de gpuTriIndices array) because multiple triangles per node can potentially be hit
		
		hitIndex = gpuTriIndices[hitIndex]; 
	}

	hitTriIdx = hitIndex;
	hitdistance =  hitT;
}


__device__ void intersectBVHandTriangles(const float4 rayorig, const float4 raydir,
	int& hitTriIdx, float& hitdistance, int& debugbingo, Vec3f& trinormal, int leafcount, int tricount, bool anyHit)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	// global threadId, see richiesams blogspot
	int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	///////////////////////////////////////////
	//// KEPLER KERNEL
	///////////////////////////////////////////

	// BVH layout Compact2 for Kepler
	int traversalStack[STACK_SIZE];

	// Live state during traversal, stored in registers.

	int		rayidx;		// not used, can be removed
	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / ray direction
	float   oodx, oody, oodz;       // ray origin / ray direction

	char*   stackPtr;               // Current position in traversal stack.
	int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
	int     nodeAddr;
	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.
	
	int threadId1; // ipv rayidx

	// Initialize (stores local variables in registers)
	{
		// Pick ray index.

		threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
		

		// Fetch ray.

		// required when tracing ray batches
		// float4 o = rays[rayidx * 2 + 0];  
		// float4 d = rays[rayidx * 2 + 1];
		//__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

		origx = rayorig.x;
		origy = rayorig.y;
		origz = rayorig.z;
		dirx = raydir.x;
		diry = raydir.y;
		dirz = raydir.z;
		tmin = rayorig.w;

		// ooeps is very small number, used instead of raydir xyz component when that component is near zero
		float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
		idirx = 1.0f / (fabsf(raydir.x) > ooeps ? raydir.x : copysignf(ooeps, raydir.x)); // inverse ray direction
		idiry = 1.0f / (fabsf(raydir.y) > ooeps ? raydir.y : copysignf(ooeps, raydir.y)); // inverse ray direction
		idirz = 1.0f / (fabsf(raydir.z) > ooeps ? raydir.z : copysignf(ooeps, raydir.z)); // inverse ray direction
		oodx = origx * idirx;  // ray origin / ray direction
		oody = origy * idiry;  // ray origin / ray direction
		oodz = origz * idirz;  // ray origin / ray direction

		// Setup traversal + initialisation

		traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
		stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
		leafAddr = 0;   // No postponed leaf.
		nodeAddr = 0;   // Start from the root.
		hitIndex = -1;  // No triangle intersected so far.
		hitT = raydir.w; // tmax  
	}

	// Traversal loop.

	while (nodeAddr != EntrypointSentinel) 
	{
		// Traverse internal nodes until all SIMD lanes have found a leaf.

		bool searchingLeaf = true; // required for warp efficiency
		while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)  
		{
			// Fetch AABBs of the two child nodes.

			// nodeAddr is an offset in number of bytes (char) in gpuNodes array
			
			float4 n0xy = tex1Dfetch(bvhNodesTexture, nodeAddr); // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)		
			float4 n1xy = tex1Dfetch(bvhNodesTexture, nodeAddr + 1); // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)		
			float4 nz = tex1Dfetch(bvhNodesTexture, nodeAddr + 2); // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)		
            float4 tmp = tex1Dfetch(bvhNodesTexture, nodeAddr + 3); // contains indices to 2 childnodes in case of innernode, see below
            int2 cnodes = *(int2*)&tmp; // cast first two floats to int
            // (childindex = size of array during building, see CudaBVH.cpp)

			// compute ray intersections with BVH node bounding box

			/// RAY BOX INTERSECTION
			// Intersect the ray against the child nodes.

			float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
			float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
			float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
			float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
			float c0loz = nz.x   * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
			float c0hiz = nz.y   * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
			float c1loz = nz.z   * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
			float c1hiz = nz.w   * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
			float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
			float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
			float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
			float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
			float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
			float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
			float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

			// ray box intersection boundary tests:
			
			float ray_tmax = 1e20;
			bool traverseChild0 = (c0min <= c0max); // && (c0min >= tmin) && (c0min <= ray_tmax);
			bool traverseChild1 = (c1min <= c1max); // && (c1min >= tmin) && (c1min <= ray_tmax);

			// Neither child was intersected => pop stack.

			if (!traverseChild0 && !traverseChild1)   
			{
				nodeAddr = *(int*)stackPtr; // fetch next node by popping the stack 
				stackPtr -= 4; // popping decrements stackPtr by 4 bytes (because stackPtr is a pointer to char)   
			}

			// Otherwise, one or both children intersected => fetch child pointers.

			else  
			{
				// set nodeAddr equal to intersected childnode index (or first childnode when both children are intersected)
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y; 

				// Both children were intersected => push the farther one on the stack.

				if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
				{   
					if (c1min < c0min)  
						swap2(nodeAddr, cnodes.y);  
					stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
					*(int*)stackPtr = cnodes.y; // push furthest node on the stack
				}
			}

			// First leaf => postpone and continue traversal.
			// leafnodes have a negative index to distinguish them from inner nodes
			// if nodeAddr less than 0 -> nodeAddr is a leaf
			if (nodeAddr < 0 && leafAddr >= 0)  
			{
				searchingLeaf = false; // required for warp efficiency
				leafAddr = nodeAddr;  
				nodeAddr = *(int*)stackPtr;  // pops next node from stack
				stackPtr -= 4;  // decrements stackptr by 4 bytes (because stackPtr is a pointer to char)
			}

			// All SIMD lanes have found a leaf => process them.

			// to increase efficiency, check if all the threads in a warp have found a leaf before proceeding to the
			// ray/triangle intersection routine
			// this bit of code requires PTX (CUDA assembly) code to work properly

			// if (!__any(searchingLeaf)) -> "__any" keyword: if none of the threads is searching a leaf, in other words
			// if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

			//if(!__any(leafAddr >= 0))     
			//    break;

			// if (!__any(searchingLeaf))
			//	break;    /// break from while loop and go to code below, processing leaf nodes

			// NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
			// tried everything with CUDA 4.2 but always got several redundant instructions.

			unsigned int mask; // replaces searchingLeaf

			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));

			if (!mask)
				break;	
		} 

		
		///////////////////////////////////////////
		/// TRIANGLE INTERSECTION
		//////////////////////////////////////

		// Process postponed leaf nodes.

		while (leafAddr < 0)  /// if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode)
		{
			// Intersect the ray against each triangle using Sven Woop's algorithm.
			// Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
			// must be transformed to "unit triangle space", before testing for intersection

			for (int triAddr = ~leafAddr;; triAddr += 3)  // triAddr is index in triWoop array (and bitwise complement of leafAddr)
			{ // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

				// Read first 16 bytes of the triangle.
				// fetch first precomputed triangle edge
				float4 v00 = tex1Dfetch(triWoopTexture, triAddr);
				
				// End marker 0x80000000 (negative zero) => all triangles in leaf processed --> terminate
				if (__float_as_int(v00.x) == 0x80000000) 
					 break;

				// Compute and check intersection t-value (hit distance along ray).
				float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;   // Origin z
				float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);  // inverse Direction z
				float t = Oz * invDz;   
				
				if (t > tmin && t < hitT)
				{
					// Compute and check barycentric u.

					// fetch second precomputed triangle edge
					float4 v11 = tex1Dfetch(triWoopTexture, triAddr + 1);
					float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;  // Origin.x
					float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;  // Direction.x
					float u = Ox + t * Dx; /// parametric equation of a ray (intersection point)

					if (u >= 0.0f && u <= 1.0f)
					{
						// Compute and check barycentric v.

						// fetch third precomputed triangle edge
						float4 v22 = tex1Dfetch(triWoopTexture, triAddr + 2);
						float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
						float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
						float v = Oy + t*Dy;

						if (v >= 0.0f && u + v <= 1.0f)
						{
							// We've got a hit!
							// Record intersection.

							hitT = t;
							hitIndex = triAddr; // store triangle index for shading

							// Closest intersection not required => terminate.
							if (anyHit)  // only true for shadow rays
							{
								nodeAddr = EntrypointSentinel;
								break;
							}

							// compute normal vector by taking the cross product of two edge vectors
							// because of Woop transformation, only one set of vectors works
							
							//trinormal = cross(Vec3f(v22.x, v22.y, v22.z), Vec3f(v11.x, v11.y, v11.z));  // works
							trinormal = cross(Vec3f(v11.x, v11.y, v11.z), Vec3f(v22.x, v22.y, v22.z));  
						}
					}
				}
			} // end triangle intersection

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if (nodeAddr < 0)    // nodeAddr is an actual leaf when < 0
			{
				nodeAddr = *(int*)stackPtr;  // pop stack
				stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
			}
		} // end leaf/triangle intersection loop
	} // end traversal loop (AABB and triangle intersection)

	// Remap intersected triangle index, and store the result.

	if (hitIndex != -1){
		hitIndex = tex1Dfetch(triIndicesTexture, hitIndex);
		// remapping tri indices delayed until this point for performance reasons
		// (slow texture memory lookup in de triIndicesTexture) because multiple triangles per node can potentially be hit
	}

	hitTriIdx = hitIndex;
	hitdistance = hitT;
}

// union struct required for mapping pixel colours to OpenGL buffer
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__device__ Vec3f renderKernel(curandState* randstate, const float4* HDRmap, const float4* gpuNodes, const float4* gpuTriWoops, 
	const float4* gpuDebugTris, const int* gpuTriIndices, Vec3f& rayorig, Vec3f& raydir, unsigned int leafcount, unsigned int tricount) 
{
	Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
	Vec3f accucolor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
	Vec3f direct = Vec3f(0, 0, 0);

	for (int bounces = 0; bounces < 4; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

		int hitSphereIdx = -1;
		int hitTriIdx = -1;
		int bestTriIdx = -1;
		int geomtype = -1;
		float hitSphereDist = 1e20;
		float hitDistance = 1e20;
		float scene_t = 1e20;
		Vec3f objcol = Vec3f(0, 0, 0);
		Vec3f emit = Vec3f(0, 0, 0);
		Vec3f hitpoint; // intersection point
		Vec3f n; // normal
		Vec3f nl; // oriented normal
		Vec3f nextdir; // ray direction of next path segment
		Vec3f trinormal = Vec3f(0, 0, 0);
		Refl_t refltype;
		float ray_tmin = 0.00001f; // set to 0.01f when using refractive material
		float ray_tmax = 1e20;

		// intersect all triangles in the scene stored in BVH

		int debugbingo = 0;

		intersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
			bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);

		//DEBUGintersectBVHandTriangles(make_float4(rayorig.x, rayorig.y, rayorig.z, ray_tmin), make_float4(raydir.x, raydir.y, raydir.z, ray_tmax),
		//gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices, bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);


		// intersect all spheres in the scene

		// float3 required for sphere intersection (to avoid "dynamic allocation not allowed" error)
		float3 rayorig_flt3 = make_float3(rayorig.x, rayorig.y, rayorig.z);
		float3 raydir_flt3 = make_float3(raydir.x, raydir.y, raydir.z);

		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = int(numspheres); i--;)  // for all spheres in scene
			// keep track of distance from origin to closest intersection point
			if ((hitSphereDist = spheres[i].intersect(Ray(rayorig_flt3, raydir_flt3))) && hitSphereDist < scene_t && hitSphereDist > 0.01f){ 
				scene_t = hitSphereDist; hitSphereIdx = i; geomtype = 1; }

		if (hitDistance < scene_t && hitDistance > ray_tmin) // triangle hit
		{
			scene_t = hitDistance;
			hitTriIdx = bestTriIdx;
			geomtype = 2;
		}

		// sky gradient colour
		//float t = 0.5f * (raydir.y + 1.2f);
		//Vec3f skycolor = Vec3f(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3f(0.9f, 0.3f, 0.0f) * t;
		
#ifdef HDR
		// HDR 

		if (scene_t > 1e19) { // if ray misses scene, return sky

			// HDR environment map code based on Syntopia "Path tracing 3D fractals"
			// http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
			// https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/IBL-Pathtracer.frag
			// GLSL code: 
			// vec3 equirectangularMap(sampler2D sampler, vec3 dir) {
			//		dir = normalize(dir);
			//		vec2 longlat = vec2(atan(dir.y, dir.x) + RotateMap, acos(dir.z));
			//		return texture2D(sampler, longlat / vec2(2.0*PI, PI)).xyz; }

			// Convert (normalized) dir to spherical coordinates.
			float longlatX = atan2f(raydir.x, raydir.z); // Y is up, swap x for y and z for x
			longlatX = longlatX < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
			float longlatY = acosf(raydir.y); // add RotateMap at some point, see Fragmentarium
			
			// map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
			float offsetY = 0.5f;
			float u = longlatX / TWO_PI; // +offsetY;
			float v = longlatY / M_PI ; 

			// map u, v to integer coordinates
			int u2 = (int)(u * HDRwidth); //% HDRwidth;
			int v2 = (int)(v * HDRheight); // % HDRheight;

			// compute the texel index in the HDR map 
			int HDRtexelidx = u2 + v2 * HDRwidth;

			//float4 HDRcol = HDRmap[HDRtexelidx];
			float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);  // fetch from texture
			Vec3f HDRcol2 = Vec3f(HDRcol.x, HDRcol.y, HDRcol.z);

			emit = HDRcol2 * 2.0f;
			accucolor += (mask * emit); 
			return accucolor; 
		}

#endif // end of HDR

		// SPHERES:
		if (geomtype == 1){
			Sphere &hitsphere = spheres[hitSphereIdx]; // hit object with closest intersection
			hitpoint = rayorig + raydir * scene_t;  // intersection point on object
			n = Vec3f(hitpoint.x - hitsphere.pos.x, hitpoint.y - hitsphere.pos.y, hitpoint.z - hitsphere.pos.z);	// normal
			n.normalize();
			nl = dot(n, raydir) < 0 ? n : n * -1; // correctly oriented normal
			objcol = Vec3f(hitsphere.col.x, hitsphere.col.y, hitsphere.col.z);   // object colour
			emit = Vec3f(hitsphere.emi.x, hitsphere.emi.y, hitsphere.emi.z);  // object emission
			refltype = hitsphere.refl;
			accucolor += (mask * emit);
		}

		// TRIANGLES:
		if (geomtype == 2){

			//pBestTri = &pTriangles[triangle_id];
			hitpoint = rayorig + raydir * scene_t; // intersection point
					
			// float4 normal = tex1Dfetch(triNormalsTexture, pBestTriIdx);	
			n = trinormal;
			n.normalize();
			nl = dot(n, raydir) < 0 ? n : n * -1;  // correctly oriented normal
			//Vec3f colour = hitTriIdx->_colorf;
			Vec3f colour = Vec3f(0.9f, 0.3f, 0.0f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
			refltype = COAT; // objectmaterial
			objcol = colour;
			emit = Vec3f(0.0, 0.0, 0);  // object emission
			accucolor += (mask * emit);
		}

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)

		// diffuse material, based on smallpt by Kevin Beason 
		if (refltype == DIFF){

			// pick two random numbers
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			Vec3f w = nl; w.normalize();
			Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
			Vec3f v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			nextdir = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
			nextdir.normalize();

			// offset origin next path segment to prevent self intersection
			hitpoint += nl * 0.001f; // scene size dependent

			// multiply mask with colour of object
			mask *= objcol;

		} // end diffuse material

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (refltype == METAL){

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float phongexponent = 30;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			Vec3f w = raydir - n * 2.0f * dot(n, raydir); w.normalize();
			Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
			Vec3f v = cross(w, u); // v is already normalised because w and u are normalised

			// compute cosine weighted random ray direction on hemisphere 
			nextdir = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
			nextdir.normalize();

			// offset origin next path segment to prevent self intersection
			hitpoint += nl * 0.0001f;  // scene size dependent

			// multiply mask with colour of object
			mask *= objcol;
		}

		// ideal specular reflection (mirror) 
		if (refltype == SPEC){

			// compute relfected ray direction according to Snell's law
			nextdir = raydir - n * dot(n, raydir) * 2.0f;
			nextdir.normalize();

			// offset origin next path segment to prevent self intersection
			hitpoint += nl * 0.001f;

			// multiply mask with colour of object
			mask *= objcol;
		}


		// COAT material based on https://github.com/peterkutz/GPUPathTracer
		// randomly select diffuse or specular reflection
		// looks okay-ish but inaccurate (no Fresnel calculation yet)
		if (refltype == COAT){

			float rouletteRandomFloat = curand_uniform(randstate);
			float threshold = 0.05f;
			Vec3f specularColor = Vec3f(1, 1, 1);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

			if (reflectFromSurface) { // calculate perfectly specular reflection

				// Ray reflected from the surface. Trace a ray in the reflection direction.
				// TODO: Use Russian roulette instead of simple multipliers! 
				// (Selecting between diffuse sample and no sample (absorption) in this case.)

				mask *= specularColor;
				nextdir = raydir - n * 2.0f * dot(n, raydir);
				nextdir.normalize();
				
				// offset origin next path segment to prevent self intersection
				hitpoint += nl * 0.001f; // scene size dependent
			}

			else {  // calculate perfectly diffuse reflection

				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				Vec3f w = nl; w.normalize();
				Vec3f u = cross((fabs(w.x) > .1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
				Vec3f v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				nextdir = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
				nextdir.normalize();

				// offset origin next path segment to prevent self intersection
				hitpoint += nl * 0.001f;  // // scene size dependent

				// multiply mask with colour of object
				mask *= objcol;
			}
		} // end COAT

		// perfectly refractive material (glass, water)
		// set ray_tmin to 0.01 when using refractive material
		if (refltype == REFR){

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.4f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(raydir, nl);
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				nextdir = raydir - n * 2.0f * dot(n, raydir);
				nextdir.normalize();

				// offset origin next path segment to prevent self intersection
				hitpoint += nl * 0.001f; // scene size dependent
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				Vec3f tdir = raydir * nnt;
				tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
				tdir.normalize();

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.2) // reflection ray
				{
					mask *= RP;
					nextdir = raydir - n * 2.0f * dot(n, raydir);
					nextdir.normalize();

					hitpoint += nl * 0.001f; // scene size dependent
				}
				else // transmission ray
				{
					mask *= TP;
					nextdir = tdir; 
					nextdir.normalize();

					hitpoint += nl * 0.001f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		rayorig = hitpoint; 
		raydir = nextdir; 
	} // end bounces for loop

	return accucolor;
}

__global__ void PathTracingKernel(Vec3f* output, Vec3f* accumbuffer, const float4* HDRmap, const float4* gpuNodes, const float4* gpuTriWoops,
  const float4* gpuDebugTris, const int* gpuTriIndices, unsigned int framenumber, unsigned int hashedframenumber, unsigned int leafcount,
  unsigned int tricount, const Camera* cudaRendercam)
{
  // assign a CUDA thread to every pixel by using the threadIndex
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  // global threadId, see richiesams blogspot
  int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  //int pixelx = threadId % scrwidth; // pixel x-coordinate on screen
  //int pixely = threadId / scrwidth; // pixel y-coordintate on screen

  // create random number generator and initialise with hashed frame number, see RichieSams blogspot
  curandState randState; // state of the random number generator, to prevent repetition
  curand_init(hashedframenumber + threadId, 0, 0, &randState);

  Vec3f finalcol; // final pixel colour 
  finalcol = Vec3f(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	
  //Vec3f rendercampos = Vec3f(0, 0.2, 4.6f); 
  Vec3f rendercampos = Vec3f(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);

  int i = (scrheight - y - 1) * scrwidth + x; // pixel index in buffer	
  int pixelx = x; // pixel x-coordinate on screen
  int pixely = scrheight - y - 1; // pixel y-coordintate on screen

  Vec3f camdir = Vec3f(0, -0.042612, -1); camdir.normalize();
  Vec3f cx = Vec3f(scrwidth * .5135f / scrheight, 0.0f, 0.0f);  // ray direction offset along X-axis 
  Vec3f cy = (cross(cx, camdir)).normalize() * .5135f; // ray dir offset along Y-axis, .5135 is FOV angle


  for (int s = 0; s < samps; s++) {

    // compute primary ray direction
    // use camera view of current frame (transformed on CPU side) to create local orthonormal basis
    Vec3f rendercamview = Vec3f(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); rendercamview.normalize(); // view is already supposed to be normalized, but normalize it explicitly just in case.
    Vec3f rendercamup = Vec3f(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); rendercamup.normalize();
    Vec3f horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis.normalize(); // Important to normalize!
    Vec3f verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis.normalize(); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

    Vec3f middle = rendercampos + rendercamview;
    Vec3f horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
    Vec3f vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

    // anti-aliasing
    // calculate center of current pixel and add random number in X and Y dimension
    // based on https://github.com/peterkutz/GPUPathTracer 

    float jitterValueX = curand_uniform(&randState) - 0.5;
    float jitterValueY = curand_uniform(&randState) - 0.5;
    float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
    float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

    // compute pixel on screen
    Vec3f pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
    Vec3f pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

    // calculation of depth of field / camera aperture 
    // based on https://github.com/peterkutz/GPUPathTracer 

    Vec3f aperturePoint = Vec3f(0, 0, 0);

    if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.

      // generate random numbers for sampling a point on the aperture
      float random1 = curand_uniform(&randState);
      float random2 = curand_uniform(&randState);

      // randomly pick a point on the circular aperture
      float angle = TWO_PI * random1;
      float distance = cudaRendercam->apertureRadius * sqrtf(random2);
      float apertureX = cos(angle) * distance;
      float apertureY = sin(angle) * distance;

      aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
      aperturePoint = rendercampos;
    }

    // calculate ray direction of next ray in path
    Vec3f apertureToImagePlane = pointOnImagePlane - aperturePoint;
    apertureToImagePlane.normalize(); // ray direction needs to be normalised

    // ray direction
    Vec3f rayInWorldSpace = apertureToImagePlane;
    rayInWorldSpace.normalize();

    // ray origin
    Vec3f originInWorldSpace = aperturePoint;

    finalcol += renderKernel(&randState, HDRmap, gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices,
        originInWorldSpace, rayInWorldSpace, leafcount, tricount) * (1.0f / samps);
  }

  // add pixel colour to accumulation buffer (accumulates all samples) 
  accumbuffer[i] += finalcol;

  // averaged colour: divide colour by the number of calculated frames so far
  Vec3f tempcol = accumbuffer[i] / framenumber;

  Colour fcolour;
  Vec3f colour = Vec3f(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

  // convert from 96-bit to 24-bit colour + perform gamma correction
  fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);

  // store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
  output[i] = Vec3f(x, y, fcolour.c);
}

bool firstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
void cudaRender(const float4* nodes, const float4* triWoops, const float4* debugTris, const int* triInds, 
	Vec3f* outputbuf, Vec3f* accumbuf, const float4* HDRmap, const unsigned int framenumber, const unsigned int hashedframenumber, 
	const unsigned int nodeSize, const unsigned int leafnodecnt, const unsigned int tricnt, const Camera* cudaRenderCam){

	if (firstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		firstTime = false;
		
		cudaChannelFormatDesc channel0desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triIndicesTexture, triInds, &channel0desc, (tricnt * 3 + leafnodecnt) * sizeof(int));  // is tricnt wel juist??

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triWoopTexture, triWoops, &channel1desc, (tricnt * 3 + leafnodecnt) * sizeof(float4));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &bvhNodesTexture, nodes, &channel3desc, nodeSize * sizeof(float4)); 

		HDRtexture.filterMode = cudaFilterModeLinear;

		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>(); 
		cudaBindTexture(NULL, &HDRtexture, HDRmap, &channel4desc, HDRwidth * HDRheight * sizeof(float4));  // 2k map:

		printf("CudaWoopTriangles texture initialised, tri count: %d\n", tricnt);
	}

	dim3 block(16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(scrwidth / block.x, scrheight / block.y, 1);

	// Configure grid and block sizes:
	int threadsPerBlock = 256;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = ((scrwidth * scrheight) + threadsPerBlock - 1) / threadsPerBlock;
	// <<<fullBlocksPerGrid, threadsPerBlock>>>
	PathTracingKernel << <grid, block >> >(outputbuf, accumbuf, HDRmap, nodes, triWoops, debugTris, 
		triInds, framenumber, hashedframenumber, leafnodecnt, tricnt, cudaRenderCam);  // texdata, texoffsets

}
