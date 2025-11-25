#ifndef __REDUCE_DEV_CUH__
#define __REDUCE_DEV_CUH__

template <typename T>
__device__  void warpReduce(T* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  __syncwarp();
  sdata[tid] += sdata[tid + 16];
  __syncwarp();
  sdata[tid] += sdata[tid + 8];
  __syncwarp();
  sdata[tid] += sdata[tid + 4];
  __syncwarp();
  sdata[tid] += sdata[tid + 2];
  __syncwarp();
  sdata[tid] += sdata[tid + 1];
  __syncwarp();
}

template <int N, typename T>
  __device__ __forceinline__ void reduce(T* shmem, int tid)
{
  if(N >= 2048)
  {
    if (tid < 1024)
      shmem[tid]+=shmem[tid+1024];
    __syncthreads();
  }
  if(N >= 1024)
  {
    if (tid < 512)
      shmem[tid]+=shmem[tid+512];
    __syncthreads();
  }
  if(N >= 512)
  {
    if (tid < 256)
      shmem[tid]+=shmem[tid+256];
    __syncthreads();
  }
  if(N >= 256)
  {
    if (tid < 128)
      shmem[tid]+=shmem[tid+128];
    __syncthreads();
  }
  if(N >= 128)
  {
    if (tid < 64)
      shmem[tid]+=shmem[tid+64];
    __syncthreads();
  }
  if (tid < 32) warpReduce(shmem, tid);
}

template <typename T>
__device__ __forceinline__ void reduce(T* shmem, int tid, unsigned int N)
{
  for (unsigned int s=N/2; s>32; s>>=1) {
  if (tid < s)
    shmem[tid] += shmem[tid + s];
    __syncthreads();
  }
  if (tid < 32) warpReduce(shmem, tid);
}

#endif // __REDUCE_DEV_CUH__
