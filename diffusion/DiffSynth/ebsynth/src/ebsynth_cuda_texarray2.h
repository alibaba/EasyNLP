// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#ifndef EBSYNTH_CUDA_TEXARRAY2_H_
#define EBSYNTH_CUDA_TEXARRAY2_H_

#include "jzq.h"
#include "ebsynth_cuda_check.h"

#include <cuda_runtime.h>

template<int N, typename T>
struct CudaVec { };

template<> struct CudaVec<1, unsigned char> { typedef uchar1 type; };
template<> struct CudaVec<2, unsigned char> { typedef uchar2 type; };
template<> struct CudaVec<4, unsigned char> { typedef uchar4 type; };

template<> struct CudaVec<1, int> { typedef int1 type; };
template<> struct CudaVec<2, int> { typedef int2 type; };
template<> struct CudaVec<4, int> { typedef int4 type; };

template<> struct CudaVec<1, float> { typedef float1 type; };
template<> struct CudaVec<2, float> { typedef float2 type; };
template<> struct CudaVec<4, float> { typedef float4 type; };

template<typename T>
struct CudaKind { };

template<> struct CudaKind<unsigned char> { static const cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned; };
template<> struct CudaKind<int>           { static const cudaChannelFormatKind kind = cudaChannelFormatKindSigned; };
template<> struct CudaKind<float>         { static const cudaChannelFormatKind kind = cudaChannelFormatKindFloat; };

__device__ Vec<1, unsigned char> cuda2jzq(const uchar1& vec) { return Vec<1, unsigned char>(vec.x); }
__device__ Vec<2, unsigned char> cuda2jzq(const uchar2& vec) { return Vec<2, unsigned char>(vec.x, vec.y); }
__device__ Vec<4, unsigned char> cuda2jzq(const uchar4& vec) { return Vec<4, unsigned char>(vec.x, vec.y, vec.z, vec.w); }

__device__ Vec<1, int> cuda2jzq(const int1& vec) { return Vec<1, int>(vec.x); }
__device__ Vec<2, int> cuda2jzq(const int2& vec) { return Vec<2, int>(vec.x, vec.y); }
__device__ Vec<4, int> cuda2jzq(const int4& vec) { return Vec<4, int>(vec.x, vec.y, vec.z, vec.w); }

__device__ Vec<1, float> cuda2jzq(const float1& vec) { return Vec<1, float>(vec.x); }
__device__ Vec<2, float> cuda2jzq(const float2& vec) { return Vec<2, float>(vec.x, vec.y); }
__device__ Vec<4, float> cuda2jzq(const float4& vec) { return Vec<4, float>(vec.x, vec.y, vec.z, vec.w); }

#define N_LAYERS(N,M) 1+(N-1)/M

template<int N, typename T>
struct TexLayer2
{
  size_t pitch;
  void* data;
  cudaTextureObject_t texObj;

  TexLayer2(){};

  TexLayer2(int width, int height)
  {
    checkCudaError(cudaMallocPitch(&data, &pitch, width*N*sizeof(T), height));

    const int bits = 8 * sizeof(T);

    const int bitsTable[4][4] = { { bits, 0, 0, 0 },
    { bits, bits, 0, 0 },
    { -1, -1, -1, -1 },
    { bits, bits, bits, bits } };

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = data;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(bitsTable[N - 1][0],
      bitsTable[N - 1][1],
      bitsTable[N - 1][2],
      bitsTable[N - 1][3],
      CudaKind<T>::kind);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    texObj = 0;
    checkCudaError(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
  }

  Vec<N, T> __device__ operator()(int x, int y) const
  {
    return cuda2jzq(tex2D<CudaVec<N, T>::type>(texObj, x, y));
  }

  void __device__ write(int x, int y, const Vec<N, T>& value)
  {
    Vec<N, T>* ptr = (Vec<N, T>*)&((unsigned char*)data)[x*sizeof(Vec<N, T>) + y*pitch];
    *ptr = value;
  }

  void destroy()
  {
    checkCudaError( cudaDestroyTextureObject(texObj) );
    checkCudaError( cudaFree(data) );
  }
};

template<int N, typename T, int M = N<3 ? N : 4>
struct TexArray2
{
  int width;
  int height;

  TexLayer2<M, T> texLayers[N_LAYERS(N, M)];

  size_t tmp_pitch;
  void*  tmp_data;

  TexArray2() : width(0),height(0),tmp_pitch(0),tmp_data(0) { }

  TexArray2(const V2i& size)
  {
    width = size(0);
    height = size(1);

    checkCudaError(cudaMallocPitch(&tmp_data, &tmp_pitch, width*N*sizeof(T), height));

    for (int i = 0; i < N_LAYERS(N, M); ++i)
      texLayers[i] = TexLayer2<M, T>(width, height);
  }

  TexArray2(int width, int height)
  {
    this->width = width;
    this->height = height;

    checkCudaError(cudaMallocPitch(&tmp_data, &tmp_pitch, width*N*sizeof(T), height));

    for (int i = 0; i < N_LAYERS(N, M); ++i)
      texLayers[i] = TexLayer2<M, T>(width, height);
  }

  Vec<N, T> __device__ operator()(int x, int y) const
  {
    Vec<N, T> ret;
    Vec<M, T> tmp;

    for (int i = 0; i < N / M; ++i){
      tmp = texLayers[i](x, y);
      for (int j = 0; j < M; ++j)
        ret[i*M + j] = tmp[j];
    }

    if (N % M != 0){
      tmp = texLayers[N / M](x, y);
      for (int j = 0; j < N % M; ++j)
        ret[(N / M)*M + j] = tmp[j];
    }

    return ret;
  }

  void __device__ write(int x, int y, const Vec<N, T>& value)
  {
    Vec<M, T> tmp;

    for (int i = 0; i < N / M; ++i){
      for (int j = 0; j < M; ++j)
        tmp[j] = value[i*M + j];
      texLayers[i].write(x, y, tmp);
    }

    if (N % M != 0){
      for (int j = 0; j < N % M; ++j)
        tmp[j] = value[(N / M)*M + j];
      texLayers[N / M].write(x, y, tmp);
    }
  }

  V2i size() const
  {
    return V2i(width,height);
  }

  void destroy()
  {
    for (int i = 0; i < N_LAYERS(N, M); ++i)
    {
      texLayers[i].destroy();
    }

    checkCudaError( cudaFree(tmp_data) );
  }
};

template<int N, typename T, int M>
__global__ void tmpToLayers(TexArray2<N, T, M> A)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<A.width && y<A.height)
  {
    Vec<N, T>* ptr = (Vec<N, T>*)&((unsigned char*)A.tmp_data)[x*sizeof(Vec<N, T>) + y*A.tmp_pitch];
    A.write(x, y, *ptr);
  }
}

template<int N, typename T, int M>
__global__ void layersToTmp(const TexArray2<N, T, M> A)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<A.width && y<A.height)
  {
    Vec<N, T> value = A(x, y);
    Vec<N, T>* ptr = (Vec<N, T>*)&((unsigned char*)A.tmp_data)[x*sizeof(Vec<N, T>) + y*A.tmp_pitch];
    *ptr = value;
  }
}

template<int N, typename T, int M>
void copy(TexArray2<N, T, M>* out_dst,const Array2<Vec<N, T>>& src)
{
  assert(out_dst != 0);
  const TexArray2<N, T, M>& dst = *out_dst;
  assert(dst.width == src.width());
  assert(dst.height == src.height());

  const int srcWidthInBytes = src.width()*sizeof(Vec<N, T>);
  const int srcPitchInBytes = srcWidthInBytes;

  checkCudaError(cudaMemcpy2D(dst.tmp_data, dst.tmp_pitch, src.data(), srcPitchInBytes, srcWidthInBytes, src.height(), cudaMemcpyHostToDevice));

  const int numThreadsPerBlock = 16;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock, numThreadsPerBlock);
  const dim3 numBlocks = dim3((src.width() + threadsPerBlock.x) / threadsPerBlock.x,
    (src.height() + threadsPerBlock.y) / threadsPerBlock.y);

  tmpToLayers << <numBlocks, threadsPerBlock >> >(dst);
}

template<int N, typename T, int M>
void copy(TexArray2<N, T, M>* out_dst,void* src_data)
{
  assert(out_dst != 0);
  const TexArray2<N, T, M>& dst = *out_dst;

  const int srcWidthInBytes = dst.width*sizeof(Vec<N,T>);
  const int srcPitchInBytes = srcWidthInBytes;

  checkCudaError(cudaMemcpy2D(dst.tmp_data, dst.tmp_pitch, src_data, srcPitchInBytes, srcWidthInBytes, dst.height, cudaMemcpyHostToDevice));

  const int numThreadsPerBlock = 16;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock, numThreadsPerBlock);
  const dim3 numBlocks = dim3((dst.width + threadsPerBlock.x) / threadsPerBlock.x,
                              (dst.height + threadsPerBlock.y) / threadsPerBlock.y);

  tmpToLayers << <numBlocks, threadsPerBlock >> >(dst);
}

template<int N, typename T, int M>
void copy(Array2<Vec<N, T>>* out_dst, const TexArray2<N, T, M>& src)
{
  assert(out_dst != 0);
  const Array2<Vec<N, T>>& dst = *out_dst;
  assert(dst.width() == src.width);
  assert(dst.height() == src.height);

  const int numThreadsPerBlock = 16;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock, numThreadsPerBlock);
  const dim3 numBlocks = dim3((dst.width() + threadsPerBlock.x) / threadsPerBlock.x,
                              (dst.height() + threadsPerBlock.y) / threadsPerBlock.y);

  layersToTmp << <numBlocks, threadsPerBlock >> >(src);

  const int dstPitchInBytes = dst.width()*sizeof(Vec<N, T>);
  checkCudaError(cudaMemcpy2D((void*)dst.data(), dstPitchInBytes, src.tmp_data, src.tmp_pitch, src.width*N*sizeof(T), src.height, cudaMemcpyDeviceToHost));
}

template<int N, typename T, int M>
void copy(void** out_dst_data, const TexArray2<N, T, M>& src)
{
  const int numThreadsPerBlock = 16;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock, numThreadsPerBlock);
  const dim3 numBlocks = dim3((src.width + threadsPerBlock.x) / threadsPerBlock.x,
                              (src.height + threadsPerBlock.y) / threadsPerBlock.y);

  layersToTmp << <numBlocks, threadsPerBlock >> >(src);

  const int dstPitchInBytes = src.width*sizeof(Vec<N, T>);
  checkCudaError(cudaMemcpy2D((void*)*out_dst_data, dstPitchInBytes, src.tmp_data, src.tmp_pitch, src.width*N*sizeof(T), src.height, cudaMemcpyDeviceToHost));
}

#endif
