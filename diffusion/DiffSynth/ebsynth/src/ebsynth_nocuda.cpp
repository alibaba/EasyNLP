// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

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

}

int ebsynthBackendAvailableCuda()
{
  return 0;
}
