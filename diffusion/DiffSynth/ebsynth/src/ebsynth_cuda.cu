// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "ebsynth_cuda_texarray2.h"
#include "ebsynth_cuda_memarray2.h"

#include <cmath>
#include <cfloat>
#include <stdint.h>

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

typedef Vec<1,float> V1f;
typedef Array2<Vec<1,float>> A2V1f;

struct pcgState
{
  uint64_t state;
  uint64_t increment;
};

__device__ void pcgAdvance(pcgState* rng)
{
  rng->state = rng->state * 6364136223846793005ULL + rng->increment;
}

__device__ uint32_t pcgOutput(uint64_t state)
{
  return (uint32_t)(((state >> 22u) ^ state) >> ((state >> 61u) + 22u));
}

__device__ uint32_t pcgRand(pcgState* rng)
{
  uint64_t oldstate = rng->state;
  pcgAdvance(rng);
  return pcgOutput(oldstate);
}

__device__ void pcgInit(pcgState* rng,uint64_t seed,uint64_t stream)
{
  rng->state = 0U;
  rng->increment = (stream << 1u) | 1u;
  pcgAdvance(rng);
  rng->state += seed;
  pcgAdvance(rng);
}

__global__ void krnlInitRngStates(const int width,
                                  const int height,
                                  pcgState* rngStates)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<width && y<height)
  {
    const int idx = x+y*width;
    pcgInit(&rngStates[idx],1337,idx);    
  }
}

pcgState* initGpuRng(const int width,
                     const int height)
{
  pcgState* gpuRngStates;
  cudaMalloc(&gpuRngStates,width*height*sizeof(pcgState));

  const dim3 threadsPerBlock(16,16);
  const dim3 numBlocks((width+threadsPerBlock.x)/threadsPerBlock.x,
                       (height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlInitRngStates<<<numBlocks,threadsPerBlock>>>(width,height,gpuRngStates);

  return gpuRngStates;
}

template<typename FUNC>
__global__ void krnlEvalErrorPass(const int patchWidth,
                                  FUNC patchError,
                                  const TexArray2<2,int> NNF,
                                  TexArray2<1,float> E)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<NNF.width && y<NNF.height)
  {
    const V2i n = NNF(x,y);
    E.write(x,y,V1f(patchError(patchWidth,x,y,n[0],n[1],FLT_MAX)));
  }
}

void __device__ updateOmega(MemArray2<int>& Omega,const int patchWidth,const int bx,const int by,const int incdec)
{
  const int r = patchWidth/2;

  for(int oy=-r;oy<=+r;oy++)
  for(int ox=-r;ox<=+r;ox++)
  {
    const int x = bx+ox;
    const int y = by+oy;
    atomicAdd(&Omega.data[x+y*Omega.width],incdec);
    //Omega.data[x+y*Omega.width] += incdec;
  }
}

int __device__ patchOmega(const int patchWidth,const int bx,const int by,const MemArray2<int>& Omega)
{
  const int r = patchWidth/2;

  int sum = 0;

  for(int oy=-r;oy<=+r;oy++)
  for(int ox=-r;ox<=+r;ox++)
  {
    const int x = bx+ox;
    const int y = by+oy;
    sum += Omega.data[x+y*Omega.width]; /// XXX: atomic read instead ??
  }

  return sum;
}

template<typename FUNC>
__device__ void tryPatch(const  V2i& sizeA,
                         const  V2i& sizeB,
                                MemArray2<int>& Omega,
                         const  int patchWidth,
                         FUNC   patchError,
                         const  float lambda,
                         const  int ax,
                         const  int ay,
                         const  int bx,
                         const  int by,
                         V2i&   nbest,
                         float& ebest)
{
  const float omegaBest = (float(sizeA(0)*sizeA(1)) /
                           float(sizeB(0)*sizeB(1))) * float(patchWidth*patchWidth);

  const float curOcc = (float(patchOmega(patchWidth,nbest(0),nbest(1),Omega))/float(patchWidth*patchWidth))/omegaBest;
  const float newOcc = (float(patchOmega(patchWidth,      bx,      by,Omega))/float(patchWidth*patchWidth))/omegaBest;

  const float curErr = ebest;
  const float newErr = patchError(patchWidth,ax,ay,bx,by,curErr+lambda*curOcc);

  if ((newErr+lambda*newOcc) < (curErr+lambda*curOcc))
  {
    updateOmega(Omega,patchWidth,      bx,      by,+1);
    updateOmega(Omega,patchWidth,nbest(0),nbest(1),-1);
    nbest = V2i(bx,by);
    ebest = newErr;
  }
}

template<typename FUNC>
__device__ void tryNeighborsOffset(const int x,
                                   const int y,
                                   const int ox,
                                   const int oy,
                                   V2i& nbest,
                                   float& ebest,
                                   const V2i& sizeA,
                                   const V2i& sizeB,
                                         MemArray2<int>& Omega,
                                   const int patchWidth,
                                   FUNC patchError,
                                   const float lambda,
                                   const TexArray2<2,int>& NNF)
{
  const int hpw = patchWidth/2;

  const V2i on = NNF(x+ox,y+oy);
  const int nx = on(0)-ox;
  const int ny = on(1)-oy;

  if (nx>=hpw && nx<sizeB(0)-hpw &&
      ny>=hpw && ny<sizeB(1)-hpw)
  {
    tryPatch(sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,nx,ny,nbest,ebest);
  }
}

template<typename FUNC>
__global__ void krnlPropagationPass(const V2i sizeA,
                                    const V2i sizeB,
                                          MemArray2<int> Omega,
                                    const int patchWidth,
                                    FUNC  patchError,
                                    const float lambda,
                                    const int r,
                                    const TexArray2<2,int> NNF,
                                    TexArray2<2,int> NNF2,
                                    TexArray2<1,float> E,
                                    TexArray2<1,unsigned char> mask)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<sizeA(0) && y<sizeA(1))
  {
    V2i   nbest = NNF(x,y);
    float ebest = E(x,y)(0);

    if (mask(x,y)[0]==255)
    {
      tryNeighborsOffset(x,y,-r,0,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,+r,0,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,0,-r,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
      tryNeighborsOffset(x,y,0,+r,nbest,ebest,sizeA,sizeB,Omega,patchWidth,patchError,lambda,NNF);
    }

    E.write(x,y,V1f(ebest));
    NNF2.write(x,y,nbest);
  }
}

template<typename FUNC>
__device__ void tryRandomOffsetInRadius(const int r,
                                        const V2i& sizeA,
                                        const V2i& sizeB,
                                              MemArray2<int>& Omega,
                                        const int patchWidth,
                                        FUNC  patchError,
                                        const float lambda,
                                        const int x,
                                        const int y,
                                        const V2i& norg,
                                        V2i&  nbest,
                                        float& ebest,
                                        pcgState* rngState)
{
  const int hpw = patchWidth/2;

  const int xmin = max(norg(0)-r,hpw);
  const int xmax = min(norg(0)+r,sizeB(0)-1-hpw);
  const int ymin = max(norg(1)-r,hpw);
  const int ymax = min(norg(1)+r,sizeB(1)-1-hpw);

  const int nx = xmin+(pcgRand(rngState)%(xmax-xmin+1));
  const int ny = ymin+(pcgRand(rngState)%(ymax-ymin+1));

  tryPatch(sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,nx,ny,nbest,ebest);
}

/*
template<typename FUNC>
__global__ void krnlRandomSearchPass(const V2i sizeA,
                                     const V2i sizeB,
                                     MemArray2<int> Omega,
                                     const int patchWidth,
                                     FUNC  patchError,
                                     const float lambda,
                                     TexArray2<2,int> NNF,
                                     TexArray2<1,float> E,
                                     TexArray2<1,unsigned char> mask,
                                     pcgState* rngStates)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<sizeA(0) && y<sizeA(1))
  {
    if (mask(x,y)[0]==255)
    {
      V2i nbest = NNF(x,y);
      float ebest = E(x,y)(0);

      const V2i norg = nbest;

      for(int r=1;r<max(sizeB(0),sizeB(1))/2;r=r*2)
      {
        tryRandomOffsetInRadius(r,sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,norg,nbest,ebest,&rngStates[x+y*NNF.width]);
      }

      E.write(x,y,V1f(ebest));
      NNF.write(x,y,nbest);
    }
  }
}
*/

template<typename FUNC>
__global__ void krnlRandomSearchPass(const V2i sizeA,
                                     const V2i sizeB,
                                     MemArray2<int> Omega,
                                     const int patchWidth,
                                     FUNC  patchError,
                                     const float lambda,
                                     const int radius,
                                     TexArray2<2,int> NNF,
                                     TexArray2<1,float> E,
                                     TexArray2<1,unsigned char> mask,
                                     pcgState* rngStates)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<sizeA(0) && y<sizeA(1))
  {
    if (mask(x,y)[0]==255)
    {
      V2i nbest = NNF(x,y);
      float ebest = E(x,y)(0);

      const V2i norg = nbest;

      tryRandomOffsetInRadius(radius,sizeA,sizeB,Omega,patchWidth,patchError,lambda,x,y,norg,nbest,ebest,&rngStates[x+y*NNF.width]);

      E.write(x,y,V1f(ebest));
      NNF.write(x,y,nbest);
    }
  }
}

template<typename FUNC>
void patchmatchGPU(const V2i sizeA,
                   const V2i sizeB,
                   MemArray2<int>& Omega,
                   const int patchWidth,
                   FUNC patchError,
                   const float lambda,
                   const int numIters,
                   const int numThreadsPerBlock,
                   TexArray2<2,int>& NNF,
                   TexArray2<2,int>& NNF2,
                   TexArray2<1,float>& E,
                   TexArray2<1,unsigned char>& mask,
                   pcgState* rngStates)
{
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
  const dim3 numBlocks = dim3((NNF.width+threadsPerBlock.x)/threadsPerBlock.x,
                              (NNF.height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchWidth,patchError,NNF,E);

  checkCudaError(cudaDeviceSynchronize());

  for(int i=0;i<numIters;i++)
  {
    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,4,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,2,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    krnlPropagationPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,1,NNF,NNF2,E,mask); std::swap(NNF,NNF2);

    checkCudaError(cudaDeviceSynchronize());

    for(int r=1;r<max(sizeB(0),sizeB(1))/2;r=r*2)
    {
      krnlRandomSearchPass<<<numBlocks,threadsPerBlock>>>(sizeA,sizeB,Omega,patchWidth,patchError,lambda,r,NNF,E,mask,rngStates);
    }

    checkCudaError(cudaDeviceSynchronize());
  }

  krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchWidth,patchError,NNF,E);

  checkCudaError(cudaDeviceSynchronize());
}

static A2V2i nnfInitRandom(const V2i& targetSize,
                    const V2i& sourceSize,
                    const int  patchSize)
{
  A2V2i NNF(targetSize);
  const int r = patchSize/2;

  for (int i = 0; i < NNF.numel(); i++)
  {
      NNF[i] = V2i
      (
          r+(rand()%(sourceSize[0]-2*r)),
          r+(rand()%(sourceSize[1]-2*r))
      );
  }

  return NNF;
}

static A2V2i nnfUpscale(const A2V2i& NNF,
                 const int    patchSize,
                 const V2i&   targetSize,
                 const V2i&   sourceSize)
{
  A2V2i NNF2x(targetSize);

  FOR(NNF2x,x,y)
  {
    NNF2x(x,y) = NNF(clamp(x/2,0,NNF.width()-1),
                     clamp(y/2,0,NNF.height()-1))*2+V2i(x%2,y%2);
  }

  FOR(NNF2x,x,y)
  {
    const V2i nn = NNF2x(x,y);

    NNF2x(x,y) = V2i(clamp(nn(0),patchSize,sourceSize(0)-patchSize-1),
                     clamp(nn(1),patchSize,sourceSize(1)-patchSize-1));
  }

  return NNF2x;
}

template<int N, typename T, int M>
__global__ void krnlVotePlain(      TexArray2<N,T,M> target,
                              const TexArray2<N,T,M> source,
                              const TexArray2<2,int> NNF,
                              const int              patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float weight = 1.0f;
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__global__ void krnlVoteWeighted(      TexArray2<N,T,M>   target,
                                 const TexArray2<N,T,M>   source,
                                 const TexArray2<2,int>   NNF,
                                 const TexArray2<1,float> E,
                                 const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<target.width && y<target.height)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      /*
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      */
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        /*if
        (
          n[0] >= 0 && n[0] < S.width () &&
          n[1] >= 0 && n[1] < S.height()
        )*/
        {
          const float error = E(x+px,y+py)(0)/(patchSize*patchSize*N);
          const float weight = 1.0f/(1.0f+error);
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target.write(x,y,v);
  }
}

template<int N, typename T, int M>
__device__ Vec<N,T> sampleBilinear(const TexArray2<N,T,M>& I,float x,float y)
{
  const int ix = x;
  const int iy = y;

  const float s = x-ix;
  const float t = y-iy;

  // XXX: clamp!!!
  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(ix  ,iy  ))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(ix+1,iy  ))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(ix  ,iy+1))+
                  (     s)*(     t)*Vec<N,float>(I(ix+1,iy+1)));
};

template<int N, typename T, int M>
__global__ void krnlResampleBilinear(TexArray2<N,T,M> O,
                                     const TexArray2<N,T,M> I)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<O.width && y<O.height)
  {
    const float s = float(I.width)/float(O.width);
    O.write(x,y,sampleBilinear(I,s*float(x),s*float(y)));
  }
}

template<int N, typename T, int M>
__global__ void krnlEvalMask(      TexArray2<1,unsigned char> mask,
                             const TexArray2<N,T,M> style,
                             const TexArray2<N,T,M> style2,
                             const int stopThreshold)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const Vec<N,T> s  = style(x,y);
    const Vec<N,T> s2 = style2(x,y);

    int maxDiff = 0;
    for(int c=0;c<N;c++)
    {
      const int diff = std::abs(int(s[c])-int(s2[c]));
      maxDiff = diff>maxDiff ? diff:maxDiff;
    }

    const Vec<1,unsigned char> msk = maxDiff < stopThreshold ? Vec<1,unsigned char>(0) : Vec<1,unsigned char>(255);

    mask.write(x,y,msk);
  }
}

__global__ void krnlDilateMask(TexArray2<1,unsigned char> mask2,
                               const TexArray2<1,unsigned char> mask,
                               const int patchSize)
{
  const int x = blockDim.x*blockIdx.x + threadIdx.x;
  const int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x<mask.width && y<mask.height)
  {
    const int r = patchSize / 2;

    Vec<1,unsigned char> msk = Vec<1,unsigned char>(0);

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      if (mask(x+px,y+py)[0]==255) { msk = Vec<1,unsigned char>(255); }
    }

    mask2.write(x,y,msk);
  }
}

template<int N, typename T, int M>
void resampleGPU(      TexArray2<N,T,M>& O,
                 const TexArray2<N,T,M>& I)
{
  const int numThreadsPerBlock = 24;
  const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
  const dim3 numBlocks = dim3((O.width+threadsPerBlock.x)/threadsPerBlock.x,
                              (O.height+threadsPerBlock.y)/threadsPerBlock.y);

  krnlResampleBilinear<<<numBlocks,threadsPerBlock>>>(O,I);

  checkCudaError(cudaDeviceSynchronize());
}

template<int NS,int NG,typename T>
struct PatchSSD_Split
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split(const TexArray2<NS,T>& targetStyle,
                 const TexArray2<NS,T>& sourceStyle,

                 const TexArray2<NG,T>& targetGuide,
                 const TexArray2<NG,T>& sourceGuide,

                 const Vec<NS,float>&   styleWeights,
                 const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

template<int NS,int NG,typename T>
struct PatchSSD_Split_Modulation
{
  const TexArray2<NS,T> targetStyle;
  const TexArray2<NS,T> sourceStyle;

  const TexArray2<NG,T> targetGuide;
  const TexArray2<NG,T> sourceGuide;

  const TexArray2<NG,T> targetModulation;

  const Vec<NS,float> styleWeights;
  const Vec<NG,float> guideWeights;

  PatchSSD_Split_Modulation(const TexArray2<NS,T>& targetStyle,
                            const TexArray2<NS,T>& sourceStyle,

                            const TexArray2<NG,T>& targetGuide,
                            const TexArray2<NG,T>& sourceGuide,

                            const TexArray2<NG,T>& targetModulation,

                            const Vec<NS,float>&   styleWeights,
                            const Vec<NG,float>&   guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    targetModulation(targetModulation),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

   __device__ float operator()(const int   patchSize,
                               const int   tx,
                               const int   ty,
                               const int   sx,
                               const int   sy,
                               const float ebest)
  {
    const int r = patchSize/2;
    float error = 0;

    for(int py=-r;py<=+r;py++)
    {
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(tx + px,ty + py);
          const Vec<NS,T> pixSs = sourceStyle(sx + px,sy + py);
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(tx + px,ty + py);
          const Vec<NG,T> pixSg = sourceGuide(sx + px,sy + py);
          const Vec<NG,float> mult = Vec<NG,float>(targetModulation(tx + px,ty + py))/255.0f;

          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*mult[i]*diff*diff;
          }
        }
      }

      if (error>ebest) { return error; }
    }

    return error;
  }
};

static V2i pyramidLevelSize(const V2i& sizeBase,const int numLevels,const int level)
{
  return V2i(V2f(sizeBase)*std::pow(2.0f,-float(numLevels-1-level)));
}

template<int NS,int NG>
void ebsynthCuda(int    numStyleChannels,
                 int    numGuideChannels,
                 int    sourceWidth,
                 int    sourceHeight,
                 void*  sourceStyleData,
                 void*  sourceGuideData,
                 int    targetWidth,
                 int    targetHeight,
                 void*  targetGuideData,
                 void*  targetModulationData,
                 float* styleWeights,
                 float* guideWeights,
                 float  uniformityWeight,
                 int    patchSize,
                 int    voteMode,
                 int    numPyramidLevels,
                 int*   numSearchVoteItersPerLevel,
                 int*   numPatchMatchItersPerLevel,
                 int*   stopThresholdPerLevel,
                 int    extraPass3x3,
                 void*  outputNnfData,
                 void*  outputImageData)
{
  const int levelCount = numPyramidLevels;

  struct PyramidLevel
  {
    PyramidLevel() { }

    int sourceWidth;
    int sourceHeight;
    int targetWidth;
    int targetHeight;

    TexArray2<NS,unsigned char> sourceStyle;
    TexArray2<NG,unsigned char> sourceGuide;
    TexArray2<NS,unsigned char> targetStyle;
    TexArray2<NS,unsigned char> targetStyle2;
    TexArray2<1,unsigned char>  mask;
    TexArray2<1,unsigned char>  mask2;
    TexArray2<NG,unsigned char> targetGuide;
    TexArray2<NG,unsigned char> targetModulation;
    TexArray2<2,int>            NNF;
    TexArray2<2,int>            NNF2;
    TexArray2<1,float>          E;
    MemArray2<int>              Omega;
  };

  std::vector<PyramidLevel> pyramid(levelCount);
  for(int level=0;level<levelCount;level++)
  {
    const V2i levelSourceSize = pyramidLevelSize(V2i(sourceWidth,sourceHeight),levelCount,level);
    const V2i levelTargetSize = pyramidLevelSize(V2i(targetWidth,targetHeight),levelCount,level);

    pyramid[level].sourceWidth  = levelSourceSize(0);
    pyramid[level].sourceHeight = levelSourceSize(1);
    pyramid[level].targetWidth  = levelTargetSize(0);
    pyramid[level].targetHeight = levelTargetSize(1);
  }

  pyramid[levelCount-1].sourceStyle  = TexArray2<NS,unsigned char>(V2i(pyramid[levelCount-1].sourceWidth,pyramid[levelCount-1].sourceHeight));
  pyramid[levelCount-1].sourceGuide  = TexArray2<NG,unsigned char>(V2i(pyramid[levelCount-1].sourceWidth,pyramid[levelCount-1].sourceHeight));
  pyramid[levelCount-1].targetGuide  = TexArray2<NG,unsigned char>(V2i(pyramid[levelCount-1].targetWidth,pyramid[levelCount-1].targetHeight));

  copy(&pyramid[levelCount-1].sourceStyle,sourceStyleData);
  copy(&pyramid[levelCount-1].sourceGuide,sourceGuideData);
  copy(&pyramid[levelCount-1].targetGuide,targetGuideData);

  if (targetModulationData)
  {
    pyramid[levelCount-1].targetModulation = TexArray2<NG,unsigned char>(V2i(pyramid[levelCount-1].targetWidth,pyramid[levelCount-1].targetHeight));
    copy(&pyramid[levelCount-1].targetModulation,targetModulationData); 
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  bool inExtraPass = false;

  pcgState* rngStates = initGpuRng(targetWidth,targetHeight);

  for (int level=0;level<pyramid.size();level++)
  {
    if (!inExtraPass)
    {
      const V2i levelSourceSize = V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight);
      const V2i levelTargetSize = V2i(pyramid[level].targetWidth,pyramid[level].targetHeight);

      pyramid[level].targetStyle  = TexArray2<NS,unsigned char>(levelTargetSize);
      pyramid[level].targetStyle2 = TexArray2<NS,unsigned char>(levelTargetSize);
      pyramid[level].mask         = TexArray2<1,unsigned char>(levelTargetSize);
      pyramid[level].mask2        = TexArray2<1,unsigned char>(levelTargetSize);
      pyramid[level].NNF          = TexArray2<2,int>(levelTargetSize);
      pyramid[level].NNF2         = TexArray2<2,int>(levelTargetSize);
      pyramid[level].Omega        = MemArray2<int>(levelSourceSize);
      pyramid[level].E            = TexArray2<1,float>(levelTargetSize);
   
      if (level<levelCount-1)
      {
        pyramid[level].sourceStyle  = TexArray2<NS,unsigned char>(levelSourceSize);
        pyramid[level].sourceGuide  = TexArray2<NG,unsigned char>(levelSourceSize);
        pyramid[level].targetGuide  = TexArray2<NG,unsigned char>(levelTargetSize);

        resampleGPU(pyramid[level].sourceStyle,pyramid[levelCount-1].sourceStyle);
        resampleGPU(pyramid[level].sourceGuide,pyramid[levelCount-1].sourceGuide);
        resampleGPU(pyramid[level].targetGuide,pyramid[levelCount-1].targetGuide);

        if (targetModulationData)
        {
          pyramid[level].targetModulation = TexArray2<NG,unsigned char>(levelTargetSize);
          resampleGPU(pyramid[level].targetModulation,pyramid[levelCount-1].targetModulation);
        }
      }

      A2V2i cpu_NNF;
      if (level>0)
      {
        A2V2i prevLevelNNF(pyramid[level-1].targetWidth,
                           pyramid[level-1].targetHeight);

        copy(&prevLevelNNF,pyramid[level-1].NNF);

        cpu_NNF = nnfUpscale(prevLevelNNF,
                             patchSize,
                             V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                             V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight));
        
        pyramid[level-1].NNF.destroy();
      }
      else
      {
        cpu_NNF = nnfInitRandom(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                                V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                                patchSize);
      }
      copy(&pyramid[level].NNF,cpu_NNF);

      /////////////////////////////////////////////////////////////////////////
      Array2<int> cpu_Omega(pyramid[level].sourceWidth,pyramid[level].sourceHeight);

      fill(&cpu_Omega,(int)0);
      for(int ay=0;ay<cpu_NNF.height();ay++)
      for(int ax=0;ax<cpu_NNF.width();ax++)
      {
        const V2i& n = cpu_NNF(ax,ay);
        const int bx = n(0);
        const int by = n(1);

        const int r = patchSize/2;

        for(int oy=-r;oy<=+r;oy++)
        for(int ox=-r;ox<=+r;ox++)
        {
          const int x = bx+ox;
          const int y = by+oy;
          cpu_Omega(x,y) += 1;
        }
      }

      copy(&pyramid[level].Omega,cpu_Omega);
      /////////////////////////////////////////////////////////////////////////
    }

    ////////////////////////////////////////////////////////////////////////////
    {
      const int numThreadsPerBlock = 24;
      const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
      const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                  (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

      krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                   pyramid[level].sourceStyle,
                                                   pyramid[level].NNF,
                                                   patchSize);

      std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
      checkCudaError( cudaDeviceSynchronize() );
    }
    ////////////////////////////////////////////////////////////////////////////

    Array2<Vec<1,unsigned char>> cpu_mask(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight));
    fill(&cpu_mask,Vec<1,unsigned char>(255));
    copy(&pyramid[level].mask,cpu_mask);

    ////////////////////////////////////////////////////////////////////////////

    for (int voteIter=0;voteIter<numSearchVoteItersPerLevel[level];voteIter++)
    {
      Vec<NS,float> styleWeightsVec;
      for(int i=0;i<NS;i++) { styleWeightsVec[i] = styleWeights[i]; }

      Vec<NG,float> guideWeightsVec;
      for(int i=0;i<NG;i++) { guideWeightsVec[i] = guideWeights[i]; }

      const int numGpuThreadsPerBlock = 24;

      if (numPatchMatchItersPerLevel[level]>0)
      {
        if (targetModulationData)
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split_Modulation<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                       pyramid[level].sourceStyle,
                                                                       pyramid[level].targetGuide,
                                                                       pyramid[level].sourceGuide,
                                                                       pyramid[level].targetModulation,
                                                                       styleWeightsVec,
                                                                       guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
        else
        {
          patchmatchGPU(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                        pyramid[level].Omega,
                        patchSize,
                        PatchSSD_Split<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                            pyramid[level].sourceStyle,
                                                            pyramid[level].targetGuide,
                                                            pyramid[level].sourceGuide,
                                                            styleWeightsVec,
                                                            guideWeightsVec),
                        uniformityWeight,
                        numPatchMatchItersPerLevel[level],
                        numGpuThreadsPerBlock,
                        pyramid[level].NNF,
                        pyramid[level].NNF2,
                        pyramid[level].E,
                        pyramid[level].mask,
                        rngStates);
        }
      }
      else
      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if (targetModulationData)
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split_Modulation<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                                                          pyramid[level].sourceStyle,
                                                                                                          pyramid[level].targetGuide,
                                                                                                          pyramid[level].sourceGuide,
                                                                                                          pyramid[level].targetModulation,
                                                                                                          styleWeightsVec,
                                                                                                          guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        else
        {
          krnlEvalErrorPass<<<numBlocks,threadsPerBlock>>>(patchSize,
                                                           PatchSSD_Split<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                                                               pyramid[level].sourceStyle,
                                                                                               pyramid[level].targetGuide,
                                                                                               pyramid[level].sourceGuide,
                                                                                               styleWeightsVec,
                                                                                               guideWeightsVec),
                                                           pyramid[level].NNF,
                                                           pyramid[level].E);
        }
        checkCudaError( cudaDeviceSynchronize() );
      }

      {
        const int numThreadsPerBlock = 24;
        const dim3 threadsPerBlock = dim3(numThreadsPerBlock,numThreadsPerBlock);
        const dim3 numBlocks = dim3((pyramid[level].targetWidth+threadsPerBlock.x)/threadsPerBlock.x,
                                    (pyramid[level].targetHeight+threadsPerBlock.y)/threadsPerBlock.y);

        if      (voteMode==EBSYNTH_VOTEMODE_PLAIN)
        {
          krnlVotePlain<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                       pyramid[level].sourceStyle,
                                                       pyramid[level].NNF,
                                                       patchSize);
        }
        else if (voteMode==EBSYNTH_VOTEMODE_WEIGHTED)
        {
          krnlVoteWeighted<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                          pyramid[level].sourceStyle,
                                                          pyramid[level].NNF,
                                                          pyramid[level].E,
                                                          patchSize);
        }

        std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
        checkCudaError( cudaDeviceSynchronize() );

        if (voteIter<numSearchVoteItersPerLevel[level]-1)
        {
          krnlEvalMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask,
                                                      pyramid[level].targetStyle,
                                                      pyramid[level].targetStyle2,
                                                      stopThresholdPerLevel[level]);
          checkCudaError( cudaDeviceSynchronize() );

          krnlDilateMask<<<numBlocks,threadsPerBlock>>>(pyramid[level].mask2,
                                                        pyramid[level].mask,
                                                        patchSize);
          std::swap(pyramid[level].mask2,pyramid[level].mask);
          checkCudaError( cudaDeviceSynchronize() );
        }
      }
    }

    if (level==levelCount-1 && (extraPass3x3==0 || (extraPass3x3!=0 && inExtraPass)))
    {      
      if (outputNnfData!=NULL) { copy(&outputNnfData,pyramid[level].NNF); }
      copy(&outputImageData,pyramid[level].targetStyle);
    }

    if ((level<levelCount-1) ||
        (extraPass3x3==0) ||
        (extraPass3x3!=0 && inExtraPass))
    {
      pyramid[level].sourceStyle.destroy();
      pyramid[level].sourceGuide.destroy();
      pyramid[level].targetGuide.destroy();
      pyramid[level].targetStyle.destroy();
      pyramid[level].targetStyle2.destroy();
      pyramid[level].mask.destroy();
      pyramid[level].mask2.destroy();
      pyramid[level].NNF2.destroy();
      pyramid[level].Omega.destroy();
      pyramid[level].E.destroy();
      if (targetModulationData) { pyramid[level].targetModulation.destroy(); }
    }

    if (level==levelCount-1 && (extraPass3x3!=0) && !inExtraPass)
    {
      inExtraPass = true;
      level--;
      patchSize = 3;
      uniformityWeight = 0;
    }
  }

  pyramid[levelCount-1].NNF.destroy();

  checkCudaError( cudaFree(rngStates) );
}

void ebsynthRunCuda(int    numStyleChannels,
                    int    numGuideChannels,
                    int    sourceWidth,
                    int    sourceHeight,
                    void*  sourceStyleData,
                    void*  sourceGuideData,
                    int    targetWidth,
                    int    targetHeight,
                    void*  targetGuideData,
                    void*  targetModulationData,
                    float* styleWeights,
                    float* guideWeights,
                    float  uniformityWeight,
                    int    patchSize,
                    int    voteMode,
                    int    numPyramidLevels,
                    int*   numSearchVoteItersPerLevel,
                    int*   numPatchMatchItersPerLevel,
                    int*   stopThresholdPerLevel,
                    int    extraPass3x3,
                    void*  outputNnfData,
                    void*  outputImageData)
{
  void (*const dispatchEbsynth[EBSYNTH_MAX_GUIDE_CHANNELS][EBSYNTH_MAX_STYLE_CHANNELS])(int,int,int,int,void*,void*,int,int,void*,void*,float*,float*,float,int,int,int,int*,int*,int*,int,void*,void*) =
  {
    { ebsynthCuda<1, 1>, ebsynthCuda<2, 1>, ebsynthCuda<3, 1>, ebsynthCuda<4, 1>, ebsynthCuda<5, 1>, ebsynthCuda<6, 1>, ebsynthCuda<7, 1>, ebsynthCuda<8, 1> },
    { ebsynthCuda<1, 2>, ebsynthCuda<2, 2>, ebsynthCuda<3, 2>, ebsynthCuda<4, 2>, ebsynthCuda<5, 2>, ebsynthCuda<6, 2>, ebsynthCuda<7, 2>, ebsynthCuda<8, 2> },
    { ebsynthCuda<1, 3>, ebsynthCuda<2, 3>, ebsynthCuda<3, 3>, ebsynthCuda<4, 3>, ebsynthCuda<5, 3>, ebsynthCuda<6, 3>, ebsynthCuda<7, 3>, ebsynthCuda<8, 3> },
    { ebsynthCuda<1, 4>, ebsynthCuda<2, 4>, ebsynthCuda<3, 4>, ebsynthCuda<4, 4>, ebsynthCuda<5, 4>, ebsynthCuda<6, 4>, ebsynthCuda<7, 4>, ebsynthCuda<8, 4> },
    { ebsynthCuda<1, 5>, ebsynthCuda<2, 5>, ebsynthCuda<3, 5>, ebsynthCuda<4, 5>, ebsynthCuda<5, 5>, ebsynthCuda<6, 5>, ebsynthCuda<7, 5>, ebsynthCuda<8, 5> },
    { ebsynthCuda<1, 6>, ebsynthCuda<2, 6>, ebsynthCuda<3, 6>, ebsynthCuda<4, 6>, ebsynthCuda<5, 6>, ebsynthCuda<6, 6>, ebsynthCuda<7, 6>, ebsynthCuda<8, 6> },
    { ebsynthCuda<1, 7>, ebsynthCuda<2, 7>, ebsynthCuda<3, 7>, ebsynthCuda<4, 7>, ebsynthCuda<5, 7>, ebsynthCuda<6, 7>, ebsynthCuda<7, 7>, ebsynthCuda<8, 7> },
    { ebsynthCuda<1, 8>, ebsynthCuda<2, 8>, ebsynthCuda<3, 8>, ebsynthCuda<4, 8>, ebsynthCuda<5, 8>, ebsynthCuda<6, 8>, ebsynthCuda<7, 8>, ebsynthCuda<8, 8> },
    { ebsynthCuda<1, 9>, ebsynthCuda<2, 9>, ebsynthCuda<3, 9>, ebsynthCuda<4, 9>, ebsynthCuda<5, 9>, ebsynthCuda<6, 9>, ebsynthCuda<7, 9>, ebsynthCuda<8, 9> },
    { ebsynthCuda<1,10>, ebsynthCuda<2,10>, ebsynthCuda<3,10>, ebsynthCuda<4,10>, ebsynthCuda<5,10>, ebsynthCuda<6,10>, ebsynthCuda<7,10>, ebsynthCuda<8,10> },
    { ebsynthCuda<1,11>, ebsynthCuda<2,11>, ebsynthCuda<3,11>, ebsynthCuda<4,11>, ebsynthCuda<5,11>, ebsynthCuda<6,11>, ebsynthCuda<7,11>, ebsynthCuda<8,11> },
    { ebsynthCuda<1,12>, ebsynthCuda<2,12>, ebsynthCuda<3,12>, ebsynthCuda<4,12>, ebsynthCuda<5,12>, ebsynthCuda<6,12>, ebsynthCuda<7,12>, ebsynthCuda<8,12> },
    { ebsynthCuda<1,13>, ebsynthCuda<2,13>, ebsynthCuda<3,13>, ebsynthCuda<4,13>, ebsynthCuda<5,13>, ebsynthCuda<6,13>, ebsynthCuda<7,13>, ebsynthCuda<8,13> },
    { ebsynthCuda<1,14>, ebsynthCuda<2,14>, ebsynthCuda<3,14>, ebsynthCuda<4,14>, ebsynthCuda<5,14>, ebsynthCuda<6,14>, ebsynthCuda<7,14>, ebsynthCuda<8,14> },
    { ebsynthCuda<1,15>, ebsynthCuda<2,15>, ebsynthCuda<3,15>, ebsynthCuda<4,15>, ebsynthCuda<5,15>, ebsynthCuda<6,15>, ebsynthCuda<7,15>, ebsynthCuda<8,15> },
    { ebsynthCuda<1,16>, ebsynthCuda<2,16>, ebsynthCuda<3,16>, ebsynthCuda<4,16>, ebsynthCuda<5,16>, ebsynthCuda<6,16>, ebsynthCuda<7,16>, ebsynthCuda<8,16> },
    { ebsynthCuda<1,17>, ebsynthCuda<2,17>, ebsynthCuda<3,17>, ebsynthCuda<4,17>, ebsynthCuda<5,17>, ebsynthCuda<6,17>, ebsynthCuda<7,17>, ebsynthCuda<8,17> },
    { ebsynthCuda<1,18>, ebsynthCuda<2,18>, ebsynthCuda<3,18>, ebsynthCuda<4,18>, ebsynthCuda<5,18>, ebsynthCuda<6,18>, ebsynthCuda<7,18>, ebsynthCuda<8,18> },
    { ebsynthCuda<1,19>, ebsynthCuda<2,19>, ebsynthCuda<3,19>, ebsynthCuda<4,19>, ebsynthCuda<5,19>, ebsynthCuda<6,19>, ebsynthCuda<7,19>, ebsynthCuda<8,19> },
    { ebsynthCuda<1,20>, ebsynthCuda<2,20>, ebsynthCuda<3,20>, ebsynthCuda<4,20>, ebsynthCuda<5,20>, ebsynthCuda<6,20>, ebsynthCuda<7,20>, ebsynthCuda<8,20> },
    { ebsynthCuda<1,21>, ebsynthCuda<2,21>, ebsynthCuda<3,21>, ebsynthCuda<4,21>, ebsynthCuda<5,21>, ebsynthCuda<6,21>, ebsynthCuda<7,21>, ebsynthCuda<8,21> },
    { ebsynthCuda<1,22>, ebsynthCuda<2,22>, ebsynthCuda<3,22>, ebsynthCuda<4,22>, ebsynthCuda<5,22>, ebsynthCuda<6,22>, ebsynthCuda<7,22>, ebsynthCuda<8,22> },
    { ebsynthCuda<1,23>, ebsynthCuda<2,23>, ebsynthCuda<3,23>, ebsynthCuda<4,23>, ebsynthCuda<5,23>, ebsynthCuda<6,23>, ebsynthCuda<7,23>, ebsynthCuda<8,23> },
    { ebsynthCuda<1,24>, ebsynthCuda<2,24>, ebsynthCuda<3,24>, ebsynthCuda<4,24>, ebsynthCuda<5,24>, ebsynthCuda<6,24>, ebsynthCuda<7,24>, ebsynthCuda<8,24> }
  };

  if (numStyleChannels>=1 && numStyleChannels<=EBSYNTH_MAX_STYLE_CHANNELS &&
      numGuideChannels>=1 && numGuideChannels<=EBSYNTH_MAX_GUIDE_CHANNELS)
  {
    dispatchEbsynth[numGuideChannels-1][numStyleChannels-1](numStyleChannels,
                                                            numGuideChannels,
                                                            sourceWidth,
                                                            sourceHeight,
                                                            sourceStyleData,
                                                            sourceGuideData,
                                                            targetWidth,
                                                            targetHeight,
                                                            targetGuideData,
                                                            targetModulationData,
                                                            styleWeights,
                                                            guideWeights,
                                                            uniformityWeight,
                                                            patchSize,
                                                            voteMode,
                                                            numPyramidLevels,
                                                            numSearchVoteItersPerLevel,
                                                            numPatchMatchItersPerLevel,
                                                            stopThresholdPerLevel,
                                                            extraPass3x3,
                                                            outputNnfData,
                                                            outputImageData);
  }
}

int ebsynthBackendAvailableCuda()
{
  int deviceCount = -1;
  if (cudaGetDeviceCount(&deviceCount)!=cudaSuccess) { return 0; }

  for (int device=0;device<deviceCount;device++)
  {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties,device)==cudaSuccess)
    {
      if (properties.major!=9999 && properties.major>=3)
      {
        return 1;
      }
    }
  }

  return 0;
}
