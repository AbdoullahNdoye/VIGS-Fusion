#ifndef __MATRIX_TYPES_H__
#define __MATRIX_TYPES_H__

#include "cuda_runtime.h"

struct Cov3
{
  float xx, xy, xz,
    yy, yz,
    zz;
};

struct Mat33
{
  float3 rows[3];
};

#endif // __MATRIX_TYPES_H__
