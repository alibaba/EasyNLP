// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#ifndef EBSYNTH_CUDA_MEMARRAY2_H_
#define EBSYNTH_CUDA_MEMARRAY2_H_

#include "jzq.h"
#include "ebsynth_cuda_check.h"

template<typename T>
struct MemArray2
{
  T* data;
  int width;
  int height;

  MemArray2() : width(0),height(0),data(0) {};

  MemArray2(const V2i& size)
  {
    width = size(0);
    height = size(1);
    checkCudaError(cudaMalloc(&data,width*height*sizeof(T)));
  }

  MemArray2(int _width,int _height)
  {
    width = _width;
    height = _height;
    checkCudaError(cudaMalloc(&data,width*height*sizeof(T)));
  }
  /*
  int       __device__ operator()(int i,int j)
  {
    return data[i+j*width];
  }

  const int& __device__ operator()(int i,int j) const
  {
    return data[i+j*width];
  }
  */

  void destroy()
  {
    checkCudaError( cudaFree(data) );
  }
};

template<typename T>
void copy(MemArray2<T>* out_dst,const Array2<T>& src)
{
  assert(out_dst != 0);
  MemArray2<T>& dst = *out_dst;
  assert(dst.width == src.width());
  assert(dst.height == src.height());

  checkCudaError(cudaMemcpy(dst.data, src.data(), src.width()*src.height()*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy(Array2<T>* out_dst,const MemArray2<T>& src)
{
  assert(out_dst != 0);
  const Array2<T>& dst = *out_dst;
  assert(dst.width() == src.width);
  assert(dst.height() == src.height);

  checkCudaError(cudaMemcpy((void*)dst.data(),src.data, src.width*src.height*sizeof(T), cudaMemcpyDeviceToHost));
}

#endif
