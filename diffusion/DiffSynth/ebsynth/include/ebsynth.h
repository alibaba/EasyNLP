// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#ifndef EBSYNTH_H
#define EBSYNTH_H

#ifndef EBSYNTH_API
  #ifdef WIN32
    #define EBSYNTH_API __declspec(dllimport)
  #else
    #define EBSYNTH_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EBSYNTH_BACKEND_CPU         0x0001
#define EBSYNTH_BACKEND_CUDA        0x0002
#define EBSYNTH_BACKEND_AUTO        0x0000

#define EBSYNTH_MAX_STYLE_CHANNELS  8
#define EBSYNTH_MAX_GUIDE_CHANNELS  24

#define EBSYNTH_VOTEMODE_PLAIN      0x0001         // weight = 1
#define EBSYNTH_VOTEMODE_WEIGHTED   0x0002         // weight = 1/(1+error)

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend);   // returns non-zero if the specified backend is available

EBSYNTH_API
void ebsynthRun(int    ebsynthBackend,             // use BACKEND_CUDA for maximum speed, BACKEND_CPU for compatibility, or BACKEND_AUTO to auto-select

                int    numStyleChannels,
                int    numGuideChannels,

                int    sourceWidth,
                int    sourceHeight,
                void*  sourceStyleData,            // (width * height * numStyleChannels) bytes, scan-line order
                void*  sourceGuideData,            // (width * height * numGuideChannels) bytes, scan-line order

                int    targetWidth,
                int    targetHeight,
                void*  targetGuideData,            // (width * height * numGuideChannels) bytes, scan-line order
                void*  targetModulationData,       // (width * height * numGuideChannels) bytes, scan-line order; pass NULL to switch off the modulation

                float* styleWeights,               // (numStyleChannels) floats
                float* guideWeights,               // (numGuideChannels) floats

                                                   // guideError(txy,sxy,ch) = guideWeights[ch] * (targetModulation[txy][ch]/255) * (targetGuide[txy][ch]-sourceGuide[sxy][ch])^2

                float  uniformityWeight,           // reasonable values are between 500-15000, 3500 is a good default

                int    patchSize,                  // odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc.
                int    voteMode,                   // use VOTEMODE_WEIGHTED for sharper result

                int    numPyramidLevels,

                int*   numSearchVoteItersPerLevel, // how many search/vote iters to perform at each level (array of ints, coarse first, fine last)
                int*   numPatchMatchItersPerLevel, // how many Patch-Match iters to perform at each level (array of ints, coarse first, fine last)

                int*   stopThresholdPerLevel,      // stop improving pixel when its change since last iteration falls under this threshold

                int    extraPass3x3,               // perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
                
                void*  outputNnfData,              // (width * height * 2) ints, scan-line order; pass NULL to ignore
                void*  outputImageData             // (width * height * numStyleChannels) bytes, scan-line order
                );

#ifdef __cplusplus
}
#endif

#endif
