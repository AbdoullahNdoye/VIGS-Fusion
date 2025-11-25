#ifndef __MATRIX_MATH_CUH
#define __MATRIX_MATH_CUH

#include "cuda_runtime.h"
#include <cuda_utils/MatrixTypes.h>

#include <math.h>
#include <cuda_utils/vector_math.cuh>

inline __host__ __device__ Cov3 make_cov3(float xx, float xy, float xz, float yy, float yz, float zz)
{
  Cov3 res;
  res.xx = xx;
  res.xy = xy;
  res.xz = xz;
  res.yy = yy;
  res.yz = yz;
  res.zz = zz;
  return res;
}

inline __host__ __device__ bool inverse(const Cov3 &in, Cov3 &out)
{
  out.xx = in.zz*in.yy - in.yz*in.yz;
  out.xy = in.xz*in.yz - in.zz*in.xy;
  out.xz = in.xy*in.yz - in.xz*in.yy;
  out.yy = in.zz*in.xx - in.xz*in.xz;
  out.yz = in.xy*in.xz - in.xx*in.yz;
  out.zz = in.xx*in.yy - in.xy*in.xy;

  float det = in.xx*out.xx+in.xy*out.xy+in.xz*out.xz;
  if(fabs(det) > 1e-9)
  {
	out.xx /= det;
	out.xy /= det;
	out.xz /= det;
	out.yy /= det;
	out.yz /= det;
	out.zz /= det;
	return true;
  }else{
	return false;
  }
}

inline __host__ __device__ Cov3 operator+( const Cov3& a, const Cov3& b)
{
  Cov3 res;
  res.xx = a.xx+b.xx;
  res.xy = a.xy+b.xy;
  res.xz = a.xz+b.xz;
  res.yy = a.yy+b.yy;
  res.yz = a.yz+b.yz;
  res.zz = a.zz+b.zz;
  return res;
}

inline __host__ __device__ Cov3 operator-( const Cov3& a, const Cov3& b)
{
  Cov3 res;
  res.xx = a.xx-b.xx;
  res.xy = a.xy-b.xy;
  res.xz = a.xz-b.xz;
  res.yy = a.yy-b.yy;
  res.yz = a.yz-b.yz;
  res.zz = a.zz-b.zz;
  return res;
}

inline __host__ __device__ void operator+=( Cov3& a, const Cov3& b)
{
  a.xx+=b.xx;
  a.xy+=b.xy;
  a.xz+=b.xz;
  a.yy+=b.yy;
  a.yz+=b.yz;
  a.zz+=b.zz;
}

inline __host__ __device__ void operator-=( Cov3& a, const Cov3& b)
{
  a.xx-=b.xx;
  a.xy-=b.xy;
  a.xz-=b.xz;
  a.yy-=b.yy;
  a.yz-=b.yz;
  a.zz-=b.zz;
}

inline __host__ __device__ Cov3 operator*( const Cov3& a, float b)
{
  Cov3 res;
  res.xx = a.xx*b;
  res.xy = a.xy*b;
  res.xz = a.xz*b;
  res.yy = a.yy*b;
  res.yz = a.yz*b;
  res.zz = a.zz*b;
  return res;
}
inline __host__ __device__ Cov3 operator*( float b, const Cov3& a)
{
  Cov3 res;
  res.xx = b*a.xx;
  res.xy = b*a.xy;
  res.xz = b*a.xz;
  res.yy = b*a.yy;
  res.yz = b*a.yz;
  res.zz = b*a.zz;
  return res;
}

inline __host__ __device__ Cov3 operator/( const Cov3& a, float b)
{
  Cov3 res;
  res.xx = a.xx/b;
  res.xy = a.xy/b;
  res.xz = a.xz/b;
  res.yy = a.yy/b;
  res.yz = a.yz/b;
  res.zz = a.zz/b;
  return res;
}

inline __host__ __device__ void operator*=( Cov3& a, float b)
{
  a.xx*=b;
  a.xy*=b;
  a.xz*=b;
  a.yy*=b;
  a.yz*=b;
  a.zz*=b;
}
inline __host__ __device__ void operator/=( Cov3& a, float b)
{
  a.xx/=b;
  a.xy/=b;
  a.xz/=b;
  a.yy/=b;
  a.yz/=b;
  a.zz/=b;
}


inline __host__ __device__ float3 operator*( const Cov3& m, const float3& b)
{
  return make_float3(m.xx * b.x + m.xy * b.y + m.xz*b.z,
					 m.xy * b.x + m.yy * b.y + m.yz*b.z,
					 m.xz * b.x + m.yz * b.y + m.zz*b.z);
}

// inline __host__ __device__ Cov3 operator*( const Cov3& a, const Cov3& b)
// {
  
//   Cov3 res;
//   res.xx = a.xx*b.xx + a.xy*b.xy + a.xz*b.xz;
//   res.xy = a.xx*b.xy + a.xy*b.yy + a.xz*b.yz;
//   res.xz = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;
//   res.yy = a.xy*b.xy + a.yy*b.yy + a.yz*b.yz;
//   res.yz = a.xy*b.xz + a.yy*b.yz + a.yz*b.zz;
//   res.zz = a.xz*b.xz + a.yz*b.yz + a.zz*b.zz;
//   return res;
// }

inline __host__ __device__ Cov3 square( const Cov3& a)
{
  Cov3 res;
  res.xx = a.xx*a.xx + a.xy*a.xy + a.xz*a.xz;
  res.xy = a.xx*a.xy + a.xy*a.yy + a.xz*a.yz;
  res.xz = a.xx*a.xz + a.xy*a.yz + a.xz*a.zz;
  res.yy = a.xy*a.xy + a.yy*a.yy + a.yz*a.yz;
  res.yz = a.xy*a.xz + a.yy*a.yz + a.yz*a.zz;
  res.zz = a.xz*a.xz + a.yz*a.yz + a.zz*a.zz;
  return res;
}

inline __host__ __device__ Mat33 operator*( const Cov3& a, const Cov3& b)
{
  Mat33 res;
  res.rows[0].x = a.xx*b.xx + a.xy*b.xy + a.xz*b.xz;
  res.rows[0].y = a.xx*b.xy + a.xy*b.yy + a.xz*b.yz;
  res.rows[0].z = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;
  res.rows[1].x = a.xy*b.xx + a.yy*b.xy + a.yz*b.xz;
  res.rows[1].y = a.xy*b.xy + a.yy*b.yy + a.yz*b.yz;
  res.rows[1].z = a.xy*b.xz + a.yy*b.yz + a.yz*b.zz;
  res.rows[2].x = a.xz*b.xx + a.yz*b.xy + a.zz*b.xz;
  res.rows[2].y = a.xz*b.xy + a.yz*b.yy + a.zz*b.yz;
  res.rows[2].z = a.xz*b.xz + a.yz*b.yz + a.zz*b.zz;

  return res;
}

inline __host__ __device__ Cov3 outer_product( const float3& v)
{
  Cov3 res;
  res.xx = v.x*v.x;
  res.xy = v.x*v.y;
  res.xz = v.x*v.z;
  res.yy = v.y*v.y;
  res.yz = v.y*v.z;
  res.zz = v.z*v.z;
  return res;
}

inline __host__ __device__ Cov3 outer_product( const float3& u, const float3& v)
{
  Cov3 res;
  res.xx = u.x*v.x;
  res.xy = u.x*v.y;
  res.xz = u.x*v.z;
  res.yy = u.y*v.y;
  res.yz = u.y*v.z;
  res.zz = u.z*v.z;
  return res;
}

inline __host__ __device__ float trace( const Cov3& a)
{
  return a.xx+a.yy+a.zz;
}

inline __host__ __device__ Mat33 make_mat33(float3 e1, float3 e2, float3 e3)
{
  Mat33 res;
  res.rows[0] = e1;
  res.rows[1] = e2;
  res.rows[2] = e3;
  return res;
}

inline __host__ __device__ Mat33 make_mat33(float m11, float m12, float m13,
                                            float m21, float m22, float m23,
                                            float m31, float m32, float m33)
{
  return make_mat33(make_float3(m11, m12, m13),
                    make_float3(m21, m22, m23),
                    make_float3(m31, m32, m33));
}


inline __host__ __device__ Mat33 operator+( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]+b.rows[i];
  }
  return res;
}

inline __host__ __device__ Mat33 operator-( Mat33& a)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = -a.rows[i];
  }
  return res;
}

inline __host__ __device__ Mat33 operator-( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]-b.rows[i];
  }
  return res;
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const float b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]*b;
  }
  return res;
}

inline __host__ __device__ Mat33 operator*( const float b, const Mat33& a)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = b*a.rows[i];
  }
  return res;
}

inline __host__ __device__ Mat33 operator/( const Mat33& a, const float b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]/b;
  }
  return res;
}

inline __host__ __device__ void operator+=( Mat33& a, const Mat33& b)
{
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] += b.rows[i];
  }
}

inline __host__ __device__ void operator-=( Mat33& a, const Mat33& b)
{
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] -= b.rows[i];
  }
}

inline __host__ __device__ void operator*=( Mat33& a, const float b)
{
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] *= b;
  }
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = make_float3(a.rows[i].x*b.rows[0].x+a.rows[i].y*b.rows[1].x+a.rows[i].z*b.rows[2].x,
                              a.rows[i].x*b.rows[0].y+a.rows[i].y*b.rows[1].y+a.rows[i].z*b.rows[2].y,
                              a.rows[i].x*b.rows[0].z+a.rows[i].y*b.rows[1].z+a.rows[i].z*b.rows[2].z);
  }
  return res;
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const Cov3& b)
{
  Mat33 res;
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = make_float3(a.rows[i].x*b.xx+a.rows[i].y*b.xy+a.rows[i].z*b.xz,
                              a.rows[i].x*b.xy+a.rows[i].y*b.yy+a.rows[i].z*b.yz,
                              a.rows[i].x*b.xz+a.rows[i].y*b.yz+a.rows[i].z*b.zz);
  }
  return res;
}

inline __host__ __device__ Mat33 operator*( const Cov3& a, const Mat33& b)
{
  Mat33 res;
  res.rows[0] = make_float3(a.xx*b.rows[0].x+a.xy*b.rows[1].x+a.xz*b.rows[2].x,
                            a.xx*b.rows[0].y+a.xy*b.rows[1].y+a.xz*b.rows[2].y,
                            a.xx*b.rows[0].z+a.xy*b.rows[1].z+a.xz*b.rows[2].z);
  res.rows[0] = make_float3(a.xy*b.rows[0].x+a.yy*b.rows[1].x+a.yz*b.rows[2].x,
                            a.xy*b.rows[0].y+a.yy*b.rows[1].y+a.yz*b.rows[2].y,
                            a.xy*b.rows[0].z+a.yy*b.rows[1].z+a.yz*b.rows[2].z);
  res.rows[0] = make_float3(a.xz*b.rows[0].x+a.yz*b.rows[1].x+a.zz*b.rows[2].x,
                            a.xz*b.rows[0].y+a.yz*b.rows[1].y+a.zz*b.rows[2].y,
                            a.xz*b.rows[0].z+a.yz*b.rows[1].z+a.zz*b.rows[2].z);
  return res;
}

inline __host__ __device__ Cov3 mult_ABAt( const Mat33& A, const Cov3& B)
{
  float3 r1 = make_float3(B.xx, B.xy, B.xz);
  float3 r2 = make_float3(B.xy, B.yy, B.yz);
  float3 r3 = make_float3(B.xz, B.yz, B.zz);

  Mat33 BAtt = make_mat33(dot(r1, A.rows[0]), dot(r2, A.rows[0]), dot(r3, A.rows[0]),
                          dot(r1, A.rows[1]), dot(r2, A.rows[1]), dot(r3, A.rows[1]),
                          dot(r1, A.rows[2]), dot(r2, A.rows[2]), dot(r3, A.rows[2]));
  
  Cov3 res = make_cov3(dot(A.rows[0], BAtt.rows[0]),
                       dot(A.rows[0], BAtt.rows[1]),
                       dot(A.rows[0], BAtt.rows[2]),
                       dot(A.rows[1], BAtt.rows[1]),
                       dot(A.rows[1], BAtt.rows[2]),
                       dot(A.rows[2], BAtt.rows[2]));
  return res;
}


inline __host__ __device__ void operator*=( Mat33& a, const Mat33& b)
{
  #pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] = make_float3(a.rows[i].x*b.rows[0].x+a.rows[i].y*b.rows[1].x+a.rows[i].z*b.rows[2].x,
                            a.rows[i].x*b.rows[0].y+a.rows[i].y*b.rows[1].y+a.rows[i].z*b.rows[2].y,
                            a.rows[i].x*b.rows[0].z+a.rows[i].y*b.rows[1].z+a.rows[i].z*b.rows[2].z);
  }
}

inline __host__ __device__ float3 operator*( const Mat33& a, const float3& b)
{
  return make_float3(dot(a.rows[0], b),
                     dot(a.rows[1], b),
                     dot(a.rows[2], b));
}

inline __host__ __device__ float3 operator*( const float3& a, const Mat33& b)
{
  return make_float3(a.x*b.rows[0].x+a.y*b.rows[1].x+a.z*b.rows[2].x,
                     a.x*b.rows[0].y+a.y*b.rows[1].y+a.z*b.rows[2].y,
                     a.x*b.rows[0].z+a.y*b.rows[1].z+a.z*b.rows[2].z);
}

inline __host__ __device__ Mat33 transpose( const Mat33& a )
{
  Mat33 res;
  res.rows[0].x = a.rows[0].x; res.rows[0].y = a.rows[1].x; res.rows[0].z = a.rows[2].x;
  res.rows[1].x = a.rows[0].y; res.rows[1].y = a.rows[1].y; res.rows[1].z = a.rows[2].y;
  res.rows[2].x = a.rows[0].z; res.rows[2].y = a.rows[1].z; res.rows[2].z = a.rows[2].z;
  return res;
}

inline __host__ __device__ float trace( const Mat33& a )
{
  return a.rows[0].x+a.rows[1].y+a.rows[2].z;
}


#endif // __MATRIX_MATH_CUH
