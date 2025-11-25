#ifndef __CUDA_UTILS_DEV_CUH__
#define __CUDA_UTILS_DEV_CUH__

#include <cuda.h>

#if CUDA_VERSION >= 9000
#include <cooperative_groups.h>
#endif

#if CUDA_VERSION>=9000
inline __device__ int atomicAggInc(int *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  int warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

inline __device__ int atomicAggDec(int *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  int warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicSub(ctr, g.size());
  return g.shfl(warp_res, 0) - g.thread_rank();
}

inline __device__ unsigned int atomicAggInc(unsigned int *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  unsigned int warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

inline __device__ unsigned int atomicAggDec(unsigned int *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  unsigned int warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicSub(ctr, g.size());
  return g.shfl(warp_res, 0) - g.thread_rank();
}
#else
inline __device__ int atomicAggInc(int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & __lanemask_lt());
  int warp_res;
  if(rank == 0)
    warp_res = atomicAdd(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res + rank;
}

inline __device__ int atomicAggDec(int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & __lanemask_lt());
  int warp_res;
  if(rank == 0)
    warp_res = atomicSub(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res - rank;
}

inline __device__ unsigned int atomicAggInc(unsigned int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & __lanemask_lt());
  unsigned int warp_res;
  if(rank == 0)
    warp_res = atomicAdd(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res + rank;
}

inline __device__ unsigned int atomicAggDec(unsigned int *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & __lanemask_lt());
  unsigned int warp_res;
  if(rank == 0)
    warp_res = atomicSub(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res - rank;
}
#endif //CUDA_VERSION>=9000


//non atomic Inc / Cec
#if CUDA_VERSION>=9000
template<typename T>
inline __device__ T aggInc(T *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  T warp_res;
  if(g.thread_rank() == 0)
  {
      *ctr+=g.size();
      warp_res = *ctr;
  }
  return g.shfl(warp_res, 0) + g.thread_rank();
}
template<typename T>
inline __device__ T aggDec(T *ctr) {
  cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  T warp_res;
  if(g.thread_rank() == 0)
  {
      *ctr-=g.size();
      warp_res = *ctr;
  }
  return g.shfl(warp_res, 0) - g.thread_rank();
}

#endif //CUDA_VERSION>=9000


#endif // __CUDA_UTILS_DEV_CUH__
