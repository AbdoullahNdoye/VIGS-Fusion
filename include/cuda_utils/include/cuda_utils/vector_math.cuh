#ifndef __VECTOR_MATH_CUH__
#define __VECTOR_MATH_CUH__

#include "cuda_runtime.h"

#include <math.h>



// additional constructors
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
  a.x += b.x;
  a.y += b.y;
}

// subtraction
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
  a.x -= b.x;
  a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float2 normalize(float2 v)
{
#ifdef __CUDACC__
    float invLen = rsqrtf(dot(v, v));
#else
    float invLen = 1.0f / sqrtf(dot(v, v));
#endif
    return v * invLen;
}

// floor
inline __host__ __device__ float2 floor(const float2 v)
{
    return make_float2(floor(v.x), floor(v.y));
}


// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
  return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(const float4 &a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}


// negate
inline __host__ __device__ float3 operator-(const float3 &a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

// addition
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, const float3 &b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

// subtraction
inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, const float3 &b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s;
	a.y *= s;
    a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
#ifdef __CUDACC__
    float invLen = rsqrtf(dot(v, v));
#else
    float invLen = 1.0f / sqrtf(dot(v, v));
#endif
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// min
inline __host__ __device__ float3 min(const float3 a, const float3 b)
{
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

// max
inline __host__ __device__ float3 max(const float3 a, const float3 b)
{
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// additional constructors
inline __host__ __device__ float4 make_float4(float s)
{
  return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), 0.f);
}

// negate
inline __host__ __device__ float4 operator-(float4 &a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// addition
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// subtraction
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s;
    a.z *= s; a.w *= s;	
}

// divide
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float4 normalize(float4 v)
{
#ifdef __CUDACC__
    float invLen = rsqrtf(dot(v, v));
#else
    float invLen = 1.0f / sqrtf(dot(v, v));
#endif
    return v * invLen;
}

// floor
inline __host__ __device__ float4 floor(const float4 v)
{
  return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}




#endif // __VECTOR_MATH_CUH__
