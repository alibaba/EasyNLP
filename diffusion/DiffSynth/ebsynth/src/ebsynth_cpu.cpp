// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "jzq.h"

#include <cmath>
#include <cfloat>
#include <cstring>

#ifdef __APPLE__
  #include <dispatch/dispatch.h>
#else
  #include <omp.h>
#endif

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

A2V2i nnfInit(const V2i& sizeA,
              const V2i& sizeB,
              const int  patchWidth)
{
  A2V2i NNF(sizeA);

  for(int xy=0;xy<NNF.numel();xy++)
  {
    NNF[xy] = V2i(patchWidth+rand()%(sizeB(0)-2*patchWidth),
                  patchWidth+rand()%(sizeB(1)-2*patchWidth));
  }

  return NNF;
}

template<typename FUNC>
A2f nnfError(const A2V2i& NNF,
             const int    patchWidth,
             FUNC         patchError)
{
  A2f E(size(NNF));
  
  #pragma omp parallel for schedule(static)
  for(int y=0;y<NNF.height();y++)
  for(int x=0;x<NNF.width();x++)
  {
    E(x,y) = patchError(patchWidth,V2i(x,y),NNF(x,y),FLT_MAX);
  }
  
  return E;
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

template<int N,typename T>
void krnlVotePlain(      Array2<Vec<N,T>>&   target,
                   const Array2<Vec<N,T>>&   source,
                   const Array2<Vec<2,int>>& NNF,
                   const int                 patchSize)
{
  for(int y=0;y<target.height();y++)
  for(int x=0;x<target.width();x++)
  {
    const int r = patchSize / 2;

    Vec<N,float> sumColor = zero<Vec<N,float>>::value();
    float sumWeight = 0;

    for (int py = -r; py <= +r; py++)
    for (int px = -r; px <= +r; px++)
    {
      if
      (
        x+px >= 0 && x+px < NNF.width () &&
        y+py >= 0 && y+py < NNF.height()
      )
      {
        const V2i n = NNF(x+px,y+py)-V2i(px,py);

        if
        (
          n[0] >= 0 && n[0] < source.width () &&
          n[1] >= 0 && n[1] < source.height()
        )
        {
          const float weight = 1.0f;
          sumColor += weight*Vec<N,float>(source(n(0),n(1)));
          sumWeight += weight;
        }
      }
    }

    const Vec<N,T> v = Vec<N,T>(sumColor/sumWeight);
    target(x,y) = v;
  }
}

#if 0
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
#endif

template<int N,typename T>
Vec<N,T> sampleBilinear(const Array2<Vec<N,T>>& I,float x,float y)
{
  const int ix = x;
  const int iy = y;

  const float s = x-ix;
  const float t = y-iy;

  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(clamp(ix  ,0,I.width()-1),clamp(iy  ,0,I.height()-1)))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(clamp(ix+1,0,I.width()-1),clamp(iy  ,0,I.height()-1)))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(clamp(ix  ,0,I.width()-1),clamp(iy+1,0,I.height()-1)))+
                  (     s)*(     t)*Vec<N,float>(I(clamp(ix+1,0,I.width()-1),clamp(iy+1,0,I.height()-1))));
};

/*
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
*/

template<int N,typename T>
void resampleCPU(      Array2<Vec<N,T>>& O,
                 const Array2<Vec<N,T>>& I)
{
  const float s = float(I.width())/float(O.width());
  
  for(int y=0;y<O.height();y++)
  for(int x=0;x<O.width();x++)
  {
    O(x,y) = sampleBilinear(I,s*float(x),s*float(y));
  }
}

template<int NS,int NG,typename T>
struct PatchSSD_Split
{
  const Array2<Vec<NS,T>>& targetStyle;
  const Array2<Vec<NS,T>>& sourceStyle;

  const Array2<Vec<NG,T>>& targetGuide;
  const Array2<Vec<NG,T>>& sourceGuide;

  const Vec<NS,float>& styleWeights;
  const Vec<NG,float>& guideWeights;

  PatchSSD_Split(const Array2<Vec<NS,T>>& targetStyle,
                 const Array2<Vec<NS,T>>& sourceStyle,

                 const Array2<Vec<NG,T>>& targetGuide,
                 const Array2<Vec<NG,T>>& sourceGuide,

                 const Vec<NS,float>& styleWeights,
                 const Vec<NG,float>& guideWeights)

  : targetStyle(targetStyle),sourceStyle(sourceStyle),
    targetGuide(targetGuide),sourceGuide(sourceGuide),
    styleWeights(styleWeights),guideWeights(guideWeights) {}

  float operator()(const int   patchSize,           
                   const V2i   txy,
                   const V2i   sxy,
                   const float ebest)
  {
    const int tx = txy(0);
    const int ty = txy(1);
    const int sx = sxy(0);
    const int sy = sxy(1);

    const int r = patchSize/2;
    float error = 0;
  
    if(tx-r>=0 && tx+r<targetStyle.width() &&
       ty-r>=0 && ty+r<targetStyle.height())
    {
      const T* ptrTs = (T*)&targetStyle(tx-r,ty-r);
      const T* ptrSs = (T*)&sourceStyle(sx-r,sy-r);
      const T* ptrTg = (T*)&targetGuide(tx-r,ty-r);
      const T* ptrSg = (T*)&sourceGuide(sx-r,sy-r);
      const int ofsTs = (targetStyle.width()-patchSize)*NS;
      const int ofsSs = (sourceStyle.width()-patchSize)*NS;
      const int ofsTg = (targetGuide.width()-patchSize)*NG;
      const int ofsSg = (sourceGuide.width()-patchSize)*NG;
      for(int j=0;j<patchSize;j++)
      {
        for(int i=0;i<patchSize;i++)
        {
          for(int k=0;k<NS;k++)
          {
            const float diff = *ptrTs - *ptrSs;
            error += styleWeights[k]*diff*diff;
            ptrTs++;
            ptrSs++;
          }
          for(int k=0;k<NG;k++)
          {
            const float diff = *ptrTg - *ptrSg;
            error += guideWeights[k]*diff*diff;
            ptrTg++;
            ptrSg++;
          }
        }        
        ptrTs += ofsTs;
        ptrSs += ofsSs;
        ptrTg += ofsTg;
        ptrSg += ofsSg;        
        if(error>ebest) { break; }
      }
    }
    else
    {
      for(int py=-r;py<=+r;py++)
      for(int px=-r;px<=+r;px++)
      {
        {
          const Vec<NS,T> pixTs = targetStyle(clamp(tx + px,0,targetStyle.width()-1),clamp(ty + py,0,targetStyle.height()-1));
          const Vec<NS,T> pixSs = sourceStyle(clamp(sx + px,0,sourceStyle.width()-1),clamp(sy + py,0,sourceStyle.height()-1));
          for(int i=0;i<NS;i++)
          {
            const float diff = float(pixTs[i]) - float(pixSs[i]);
            error += styleWeights[i]*diff*diff;
          }
        }

        {
          const Vec<NG,T> pixTg = targetGuide(clamp(tx + px,0,targetGuide.width()-1),clamp(ty + py,0,targetGuide.height()-1));
          const Vec<NG,T> pixSg = sourceGuide(clamp(sx + px,0,sourceGuide.width()-1),clamp(sy + py,0,sourceGuide.height()-1));
          for(int i=0;i<NG;i++)
          {
            const float diff = float(pixTg[i]) - float(pixSg[i]);
            error += guideWeights[i]*diff*diff;
          }
        }
      }
    }

    return error;
  }
};

/*
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
          const Vec<NG,float> mult = Vec<NG,float>(targetModulation(tx,ty))/255.0f;

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
*/

static V2i pyramidLevelSize(const V2i& sizeBase,const int numLevels,const int level)
{
  return V2i(V2f(sizeBase)*std::pow(2.0f,-float(numLevels-1-level)));
}

template<typename T>
void copy(Array2<T>* out_dst,void* src)
{
  Array2<T>& dst = *out_dst;
  memcpy(dst.data(),src,numel(dst)*sizeof(T));
}

template<typename T>
void copy(void** out_dst,const Array2<T>& src)
{
  void*& dst = *out_dst;
  memcpy(dst,src.data(),numel(src)*sizeof(T));
}

void updateOmega(A2i& Omega,const V2i& sizeA,const int patchWidth,const V2i& axy,const V2i& bxy,const int incdec)
{
  const int r = patchWidth/2;
  
  int* ptr = (int*)&Omega(bxy(0)-r,bxy(1)-r);
  const int ofs = (Omega.width()-patchWidth);

  for(int j=0;j<patchWidth;j++)
  {
    for(int i=0;i<patchWidth;i++)
    {
      *ptr += incdec;
      ptr++;
    }
    ptr += ofs;
  }
}

static int patchOmega(const int patchWidth,const V2i& bxy,const A2i& Omega)
{
  const int r = patchWidth/2;
  
  int sum = 0;

  const int* ptr = (int*)&Omega(bxy(0)-r,bxy(1)-r);
  const int ofs = (Omega.width()-patchWidth);

  for(int j=0;j<patchWidth;j++)
  {
    for(int i=0;i<patchWidth;i++)
    {
      sum += (*ptr);
      ptr++;
    }
    ptr += ofs;
  }

  return sum;
}

template<typename FUNC>
bool tryPatch(FUNC patchError,const V2i& sizeA,int patchWidth,const V2i& axy,const V2i& bxy,A2V2i& N,A2f& E,A2i& Omega,float omegaBest,float lambda)
{
  const float curOcc = (float(patchOmega(patchWidth,N(axy),Omega))/float(patchWidth*patchWidth))/omegaBest;
  const float newOcc = (float(patchOmega(patchWidth,   bxy,Omega))/float(patchWidth*patchWidth))/omegaBest;
    
  const float curErr = E(axy);
  const float newErr = patchError(patchWidth,axy,bxy,curErr+lambda*curOcc);

  if ((newErr+lambda*newOcc) < (curErr+lambda*curOcc))
  {
    updateOmega(Omega,sizeA,patchWidth,axy,bxy   ,+1);
    updateOmega(Omega,sizeA,patchWidth,axy,N(axy),-1);
    N(axy) = bxy;
    E(axy) = newErr;
  }

  return true;
}

template<typename FUNC>
void patchmatch(const V2i&  sizeA,
                const V2i&  sizeB,
                const int   patchWidth,
                FUNC        patchError,
                const float lambda,
                const int   numIters,
                const int   numThreads,
                A2V2i& N,
                A2f&   E,
                A2i&   Omega)
{
  const int w = patchWidth;
    
  E = nnfError(N,patchWidth,patchError);
  
  const float sra = 0.5f;
  
  std::vector<int> irad;
  
  irad.push_back((sizeB(0) > sizeB(1) ? sizeB(0) : sizeB(1)));
  
  while (irad.back() != 1) irad.push_back(int(std::pow(sra, int(irad.size())) * irad[0]));
  
  const int nir = int(irad.size());
  
#ifdef __APPLE__
  dispatch_queue_t gcdq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH,0);
  const int numThreads_ = 8;
#else
  const int numThreads_ = numThreads<1 ? omp_get_max_threads() : numThreads;
#endif

  const int minTileHeight = 8;
  const int numTiles = int(ceil(float(sizeA(1))/float(numThreads_))) > minTileHeight ? numThreads_ : std::max(int(ceil(float(sizeA(1))/float(minTileHeight))),1);
  const int tileHeight = sizeA(1)/numTiles;

  const float omegaBest = (float(sizeA(0)*sizeA(1)) /
                           float(sizeB(0)*sizeB(1))) * float(patchWidth*patchWidth);

  fill(&Omega,(int)0);
  for(int y=0;y<sizeA(1);y++)
  for(int x=0;x<sizeA(0);x++)
  {
    updateOmega(Omega,sizeA,w,V2i(x,y),N(x,y),+1);
  }

  for (int iter = 0; iter < numIters; iter++)
  {
    const int iter_seed = rand();
    
#ifdef __APPLE__
    dispatch_apply(numTiles,gcdq,^(size_t blockIdx)
#else
    #pragma omp parallel num_threads(numTiles)
#endif
    {
      const bool odd = (iter%2 == 0);
      
#ifdef __APPLE__
      const int threadId = blockIdx;
#else
      const int threadId = omp_get_thread_num();
#endif

      const int _y0 = threadId*tileHeight;
      const int _y1 = threadId==numTiles-1 ? sizeA(1) : std::min(_y0+tileHeight,sizeA(1));
      
      const int q  = odd ? 1 : -1;
      const int x0 = odd ? 0 : sizeA(0)-1;
      const int y0 = odd ? _y0 : _y1-1;
      const int x1 = odd ? sizeA(0) : -1;
      const int y1 = odd ? _y1 : _y0-1;
      
      for (int y = y0; y != y1; y += q)
      for (int x = x0; x != x1; x += q)
      {        
        if (odd ? (x > 0) : (x < sizeA(0)-1))
        {
          V2i n = N(x-q,y); n[0] += q;
          
          if (odd ? (n[0] < sizeB(0)-w/2) : (n[0] >= w/2))
          {
            tryPatch(patchError,sizeA,w,V2i(x,y),n,N,E,Omega,omegaBest,lambda);
          }
        }
        
        if (odd ? (y > 0) : (y <sizeA(1)-1))
        {
          V2i n = N(x,y-q); n[1] += q;
          
          if (odd ? (n[1] < sizeB(1)-w/2) : (n[1] >= w/2))
          {
            tryPatch(patchError,sizeA,w,V2i(x,y),n,N,E,Omega,omegaBest,lambda);
          }
        }
           
        #define RANDI(u) (18000 * ((u) & 65535) + ((u) >> 16))

        unsigned int seed = (x | (y<<11)) ^ iter_seed;
        seed = RANDI(seed);
      
        const V2i pix0 = N(x,y);
        //for (int i = 0; i < nir; i++)
        for (int i = nir-1; i >=0; i--)
        {
          V2i tl = pix0 - V2i(irad[i], irad[i]);
          V2i br = pix0 + V2i(irad[i], irad[i]);
          
          tl = std::max(tl,V2i(w/2,w/2));
          br = std::min(br,sizeB-V2i(w/2,w/2));
          
          const int _rndX = RANDI(seed);
          const int _rndY = RANDI(_rndX);
          seed=_rndY;
          
          const V2i n = V2i
          (
            tl[0] + (_rndX % (br[0]-tl[0])),
            tl[1] + (_rndY % (br[1]-tl[1]))
          );
        
          tryPatch(patchError,sizeA,w,V2i(x,y),n,N,E,Omega,omegaBest,lambda);
        }

        #undef RANDI
      }
    } 
#ifdef __APPLE__
    );
#endif
  }
}

template<int NS,int NG>
void ebsynthCpu(int    numStyleChannels,
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

    Array2<Vec<NS,unsigned char>> sourceStyle;
    Array2<Vec<NG,unsigned char>> sourceGuide;
    Array2<Vec<NS,unsigned char>> targetStyle;
    Array2<Vec<NS,unsigned char>> targetStyle2;
    //Array2<unsigned char>         mask;
    //Array2<unsigned char>         mask2;
    Array2<Vec<NG,unsigned char>> targetGuide;
    Array2<Vec<NG,unsigned char>> targetModulation;
    Array2<Vec<2,int>>            NNF;
    //Array2<Vec<2,int>>            NNF2;
    Array2<float>                 E;
    Array2<int>                   Omega;
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

  pyramid[levelCount-1].sourceStyle  = Array2<Vec<NS,unsigned char>>(V2i(pyramid[levelCount-1].sourceWidth,pyramid[levelCount-1].sourceHeight));
  pyramid[levelCount-1].sourceGuide  = Array2<Vec<NG,unsigned char>>(V2i(pyramid[levelCount-1].sourceWidth,pyramid[levelCount-1].sourceHeight));
  pyramid[levelCount-1].targetGuide  = Array2<Vec<NG,unsigned char>>(V2i(pyramid[levelCount-1].targetWidth,pyramid[levelCount-1].targetHeight));

  copy(&pyramid[levelCount-1].sourceStyle,sourceStyleData);
  copy(&pyramid[levelCount-1].sourceGuide,sourceGuideData);
  copy(&pyramid[levelCount-1].targetGuide,targetGuideData);

  if (targetModulationData)
  {
    pyramid[levelCount-1].targetModulation = Array2<Vec<NG,unsigned char>>(V2i(pyramid[levelCount-1].targetWidth,pyramid[levelCount-1].targetHeight));
    copy(&pyramid[levelCount-1].targetModulation,targetModulationData); 
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  bool inExtraPass = false;

  for (int level=0;level<pyramid.size();level++)
  {
    if (!inExtraPass)
    {
      const V2i levelSourceSize = V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight);
      const V2i levelTargetSize = V2i(pyramid[level].targetWidth,pyramid[level].targetHeight);

      pyramid[level].targetStyle  = Array2<Vec<NS,unsigned char>>(levelTargetSize);
      pyramid[level].targetStyle2 = Array2<Vec<NS,unsigned char>>(levelTargetSize);
      //pyramid[level].mask         = Array2<unsigned char>(levelTargetSize);
      //pyramid[level].mask2        = Array2<unsigned char>(levelTargetSize);
      pyramid[level].NNF          = Array2<Vec<2,int>>(levelTargetSize);
      //pyramid[level].NNF2         = Array2<Vec<2,int>>(levelTargetSize);
      pyramid[level].Omega        = Array2<int>(levelSourceSize);
      pyramid[level].E            = Array2<float>(levelTargetSize);
   
      if (level<levelCount-1)
      {
        pyramid[level].sourceStyle  = Array2<Vec<NS,unsigned char>>(levelSourceSize);
        pyramid[level].sourceGuide  = Array2<Vec<NG,unsigned char>>(levelSourceSize);
        pyramid[level].targetGuide  = Array2<Vec<NG,unsigned char>>(levelTargetSize);

        resampleCPU(pyramid[level].sourceStyle,pyramid[levelCount-1].sourceStyle);
        resampleCPU(pyramid[level].sourceGuide,pyramid[levelCount-1].sourceGuide);
        resampleCPU(pyramid[level].targetGuide,pyramid[levelCount-1].targetGuide);

        if (targetModulationData)
        {
          resampleCPU(pyramid[level].targetModulation,pyramid[levelCount-1].targetModulation);
          pyramid[level].targetModulation = Array2<Vec<NG,unsigned char>>(levelTargetSize);
        }
      }

      A2V2i cpu_NNF;
      if (level>0)
      {
        pyramid[level].NNF = nnfUpscale(pyramid[level-1].NNF,
                                        patchSize,
                                        V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                                        V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight));
        
        pyramid[level-1].NNF = A2V2i();
      }
      else
      {
        pyramid[level].NNF = nnfInitRandom(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                                           V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                                           patchSize);
      }

      /////////////////////////////////////////////////////////////////////////
      /*
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
      */
      /////////////////////////////////////////////////////////////////////////
    }

    ////////////////////////////////////////////////////////////////////////////
    {
      krnlVotePlain(pyramid[level].targetStyle2,
                    pyramid[level].sourceStyle,
                    pyramid[level].NNF,
                    patchSize);

      std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);
    }
    ////////////////////////////////////////////////////////////////////////////

    //Array2<Vec<1,unsigned char>> cpu_mask(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight));
    //fill(&cpu_mask,Vec<1,unsigned char>(255));
    //copy(&pyramid[level].mask,cpu_mask);

    ////////////////////////////////////////////////////////////////////////////

    for (int voteIter=0;voteIter<numSearchVoteItersPerLevel[level];voteIter++)
    {
      Vec<NS,float> styleWeightsVec;
      for(int i=0;i<NS;i++) { styleWeightsVec[i] = styleWeights[i]; }

      Vec<NG,float> guideWeightsVec;
      for(int i=0;i<NG;i++) { guideWeightsVec[i] = guideWeights[i]; }

      //if (numPatchMatchItersPerLevel[level]>0)
      {
        /*if (targetModulationData)
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
        else*/
        {
          patchmatch(V2i(pyramid[level].targetWidth,pyramid[level].targetHeight),
                     V2i(pyramid[level].sourceWidth,pyramid[level].sourceHeight),
                     patchSize,
                     PatchSSD_Split<NS,NG,unsigned char>(pyramid[level].targetStyle,
                                                         pyramid[level].sourceStyle,
                                                         pyramid[level].targetGuide,
                                                         pyramid[level].sourceGuide,
                                                         styleWeightsVec,
                                                         guideWeightsVec),
                     uniformityWeight,                             
                     numPatchMatchItersPerLevel[level],
                     -1,
                     pyramid[level].NNF,
                     pyramid[level].E,
                     pyramid[level].Omega);
        }
      }
      /*
      else
      {       
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
      */
      {
        //if      (voteMode==EBSYNTH_VOTEMODE_PLAIN)
        {
          krnlVotePlain(pyramid[level].targetStyle2,
                        pyramid[level].sourceStyle,
                        pyramid[level].NNF,
                        patchSize);
        }
        /*else if (voteMode==EBSYNTH_VOTEMODE_WEIGHTED)
        {
          krnlVoteWeighted<<<numBlocks,threadsPerBlock>>>(pyramid[level].targetStyle2,
                                                          pyramid[level].sourceStyle,
                                                          pyramid[level].NNF,
                                                          pyramid[level].E,
                                                          patchSize);
        }*/

        std::swap(pyramid[level].targetStyle2,pyramid[level].targetStyle);

        /*
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
        */
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
      pyramid[level].sourceStyle = Array2<Vec<NS,unsigned char>>();
      pyramid[level].sourceGuide = Array2<Vec<NG,unsigned char>>();
      pyramid[level].targetGuide = Array2<Vec<NG,unsigned char>>();
      pyramid[level].targetStyle = Array2<Vec<NS,unsigned char>>();
      pyramid[level].targetStyle2 = Array2<Vec<NS,unsigned char>>();
      //pyramid[level].mask = Array2<unsigned char>();
      //pyramid[level].mask2 = Array2<unsigned char>();
      //pyramid[level].NNF2 = Array2<Vec<2,int>>();
      pyramid[level].Omega = Array2<int>();
      pyramid[level].E = Array2<float>();
      if (targetModulationData) { pyramid[level].targetModulation = Array2<Vec<NG,unsigned char>>(); }
    }

    if (level==levelCount-1 && (extraPass3x3!=0) && !inExtraPass)
    {
      inExtraPass = true;
      level--;
      patchSize = 3;
      uniformityWeight = 0;
    }
  }

  pyramid[levelCount-1].NNF = Array2<Vec<2,int>>();
}

void ebsynthRunCpu(int    numStyleChannels,
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
    { ebsynthCpu<1, 1>, ebsynthCpu<2, 1>, ebsynthCpu<3, 1>, ebsynthCpu<4, 1>, ebsynthCpu<5, 1>, ebsynthCpu<6, 1>, ebsynthCpu<7, 1>, ebsynthCpu<8, 1> },
    { ebsynthCpu<1, 2>, ebsynthCpu<2, 2>, ebsynthCpu<3, 2>, ebsynthCpu<4, 2>, ebsynthCpu<5, 2>, ebsynthCpu<6, 2>, ebsynthCpu<7, 2>, ebsynthCpu<8, 2> },
    { ebsynthCpu<1, 3>, ebsynthCpu<2, 3>, ebsynthCpu<3, 3>, ebsynthCpu<4, 3>, ebsynthCpu<5, 3>, ebsynthCpu<6, 3>, ebsynthCpu<7, 3>, ebsynthCpu<8, 3> },
    { ebsynthCpu<1, 4>, ebsynthCpu<2, 4>, ebsynthCpu<3, 4>, ebsynthCpu<4, 4>, ebsynthCpu<5, 4>, ebsynthCpu<6, 4>, ebsynthCpu<7, 4>, ebsynthCpu<8, 4> },
    { ebsynthCpu<1, 5>, ebsynthCpu<2, 5>, ebsynthCpu<3, 5>, ebsynthCpu<4, 5>, ebsynthCpu<5, 5>, ebsynthCpu<6, 5>, ebsynthCpu<7, 5>, ebsynthCpu<8, 5> },
    { ebsynthCpu<1, 6>, ebsynthCpu<2, 6>, ebsynthCpu<3, 6>, ebsynthCpu<4, 6>, ebsynthCpu<5, 6>, ebsynthCpu<6, 6>, ebsynthCpu<7, 6>, ebsynthCpu<8, 6> },
    { ebsynthCpu<1, 7>, ebsynthCpu<2, 7>, ebsynthCpu<3, 7>, ebsynthCpu<4, 7>, ebsynthCpu<5, 7>, ebsynthCpu<6, 7>, ebsynthCpu<7, 7>, ebsynthCpu<8, 7> },
    { ebsynthCpu<1, 8>, ebsynthCpu<2, 8>, ebsynthCpu<3, 8>, ebsynthCpu<4, 8>, ebsynthCpu<5, 8>, ebsynthCpu<6, 8>, ebsynthCpu<7, 8>, ebsynthCpu<8, 8> },
    { ebsynthCpu<1, 9>, ebsynthCpu<2, 9>, ebsynthCpu<3, 9>, ebsynthCpu<4, 9>, ebsynthCpu<5, 9>, ebsynthCpu<6, 9>, ebsynthCpu<7, 9>, ebsynthCpu<8, 9> },
    { ebsynthCpu<1,10>, ebsynthCpu<2,10>, ebsynthCpu<3,10>, ebsynthCpu<4,10>, ebsynthCpu<5,10>, ebsynthCpu<6,10>, ebsynthCpu<7,10>, ebsynthCpu<8,10> },
    { ebsynthCpu<1,11>, ebsynthCpu<2,11>, ebsynthCpu<3,11>, ebsynthCpu<4,11>, ebsynthCpu<5,11>, ebsynthCpu<6,11>, ebsynthCpu<7,11>, ebsynthCpu<8,11> },
    { ebsynthCpu<1,12>, ebsynthCpu<2,12>, ebsynthCpu<3,12>, ebsynthCpu<4,12>, ebsynthCpu<5,12>, ebsynthCpu<6,12>, ebsynthCpu<7,12>, ebsynthCpu<8,12> },
    { ebsynthCpu<1,13>, ebsynthCpu<2,13>, ebsynthCpu<3,13>, ebsynthCpu<4,13>, ebsynthCpu<5,13>, ebsynthCpu<6,13>, ebsynthCpu<7,13>, ebsynthCpu<8,13> },
    { ebsynthCpu<1,14>, ebsynthCpu<2,14>, ebsynthCpu<3,14>, ebsynthCpu<4,14>, ebsynthCpu<5,14>, ebsynthCpu<6,14>, ebsynthCpu<7,14>, ebsynthCpu<8,14> },
    { ebsynthCpu<1,15>, ebsynthCpu<2,15>, ebsynthCpu<3,15>, ebsynthCpu<4,15>, ebsynthCpu<5,15>, ebsynthCpu<6,15>, ebsynthCpu<7,15>, ebsynthCpu<8,15> },
    { ebsynthCpu<1,16>, ebsynthCpu<2,16>, ebsynthCpu<3,16>, ebsynthCpu<4,16>, ebsynthCpu<5,16>, ebsynthCpu<6,16>, ebsynthCpu<7,16>, ebsynthCpu<8,16> },
    { ebsynthCpu<1,17>, ebsynthCpu<2,17>, ebsynthCpu<3,17>, ebsynthCpu<4,17>, ebsynthCpu<5,17>, ebsynthCpu<6,17>, ebsynthCpu<7,17>, ebsynthCpu<8,17> },
    { ebsynthCpu<1,18>, ebsynthCpu<2,18>, ebsynthCpu<3,18>, ebsynthCpu<4,18>, ebsynthCpu<5,18>, ebsynthCpu<6,18>, ebsynthCpu<7,18>, ebsynthCpu<8,18> },
    { ebsynthCpu<1,19>, ebsynthCpu<2,19>, ebsynthCpu<3,19>, ebsynthCpu<4,19>, ebsynthCpu<5,19>, ebsynthCpu<6,19>, ebsynthCpu<7,19>, ebsynthCpu<8,19> },
    { ebsynthCpu<1,20>, ebsynthCpu<2,20>, ebsynthCpu<3,20>, ebsynthCpu<4,20>, ebsynthCpu<5,20>, ebsynthCpu<6,20>, ebsynthCpu<7,20>, ebsynthCpu<8,20> },
    { ebsynthCpu<1,21>, ebsynthCpu<2,21>, ebsynthCpu<3,21>, ebsynthCpu<4,21>, ebsynthCpu<5,21>, ebsynthCpu<6,21>, ebsynthCpu<7,21>, ebsynthCpu<8,21> },
    { ebsynthCpu<1,22>, ebsynthCpu<2,22>, ebsynthCpu<3,22>, ebsynthCpu<4,22>, ebsynthCpu<5,22>, ebsynthCpu<6,22>, ebsynthCpu<7,22>, ebsynthCpu<8,22> },
    { ebsynthCpu<1,23>, ebsynthCpu<2,23>, ebsynthCpu<3,23>, ebsynthCpu<4,23>, ebsynthCpu<5,23>, ebsynthCpu<6,23>, ebsynthCpu<7,23>, ebsynthCpu<8,23> },
    { ebsynthCpu<1,24>, ebsynthCpu<2,24>, ebsynthCpu<3,24>, ebsynthCpu<4,24>, ebsynthCpu<5,24>, ebsynthCpu<6,24>, ebsynthCpu<7,24>, ebsynthCpu<8,24> }
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

int ebsynthBackendAvailableCpu()
{
  return 1;
}
