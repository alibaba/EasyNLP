// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "ebsynth_cpu.h"
#include "ebsynth_cuda.h"

#include <cstdio>
#include <cmath>

EBSYNTH_API
void ebsynthRun(int    ebsynthBackend,
                int    numStyleChannels,
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
  void (*backendDispatch)(int,int,int,int,void*,void*,int,int,void*,void*,float*,float*,float,int,int,int,int*,int*,int*,int,void*,void*) = 0;
  
  if      (ebsynthBackend==EBSYNTH_BACKEND_CPU ) { backendDispatch = ebsynthRunCpu;  }
  else if (ebsynthBackend==EBSYNTH_BACKEND_CUDA) { backendDispatch = ebsynthRunCuda; }
  else if (ebsynthBackend==EBSYNTH_BACKEND_AUTO) { backendDispatch = ebsynthBackendAvailableCuda() ? ebsynthRunCuda : ebsynthRunCpu; }
  
  if (backendDispatch!=0)
  {
    backendDispatch(numStyleChannels,
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

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend)
{
  if      (ebsynthBackend==EBSYNTH_BACKEND_CPU ) { return ebsynthBackendAvailableCpu();  }
  else if (ebsynthBackend==EBSYNTH_BACKEND_CUDA) { return ebsynthBackendAvailableCuda(); }
  else if (ebsynthBackend==EBSYNTH_BACKEND_AUTO) { return ebsynthBackendAvailableCpu() || ebsynthBackendAvailableCuda(); }
  
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cmath>

#include <vector>
#include <string>
#include <algorithm>

#include "jzq.h"

template<typename FUNC>
bool tryToParseArg(const std::vector<std::string>& args,int* inout_argi,const char* name,bool* out_fail,FUNC handler)
{
  int& argi = *inout_argi;
  bool& fail = *out_fail;

  if (argi<0 || argi>=args.size()) { fail = true; return false; }

  if (args[argi]==name)
  {
    argi++;
    fail = !handler();    
    return true;
  }

  fail = false; return false; 
}

bool tryToParseIntArg(const std::vector<std::string>& args,int* inout_argi,const char* name,int* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      const std::string& arg = args[argi];
      try
      {
        std::size_t pos = 0;
        *out_value = std::stoi(arg,&pos);
        if (pos!=arg.size()) { printf("error: bad %s argument '%s'\n",name,arg.c_str()); return false; }
        return true;
      }
      catch(...)
      {
        printf("error: bad %s argument '%s'\n",name,arg.c_str());
        return false;
      }   
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseFloatArg(const std::vector<std::string>& args,int* inout_argi,const char* name,float* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      const std::string& arg = args[argi];
      try
      {
        std::size_t pos = 0;
        *out_value = std::stof(arg,&pos);
        if (pos!=arg.size()) { printf("error: bad %s argument '%s'\n",name,arg.c_str()); return false; }
        return true;
      }
      catch(...)
      {
        printf("error: bad %s argument '%s'\n",name,args[argi].c_str());
        return false;
      }   
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseStringArg(const std::vector<std::string>& args,int* inout_argi,const char* name,std::string* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if (argi<args.size())
    {
      *out_value = args[argi];
      return true;
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

bool tryToParseStringPairArg(const std::vector<std::string>& args,int* inout_argi,const char* name,std::pair<std::string,std::string>* out_value,bool* out_fail)
{
  return tryToParseArg(args,inout_argi,name,out_fail,[&]
  {
    int& argi = *inout_argi;
    if ((argi+1)<args.size())
    {
      *out_value = std::make_pair(args[argi],args[argi+1]);
      argi++;
      return true;
    }
    printf("error: missing argument for the %s option\n",name);
    return false;
  });
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* tryLoad(const std::string& fileName,int* width,int* height)
{
  unsigned char* data = stbi_load(fileName.c_str(),width,height,NULL,4);
  if (data==NULL)
  {
    printf("error: failed to load '%s'\n",fileName.c_str());
    printf("%s\n",stbi_failure_reason());
    exit(1);
  }
  return data;
}

int evalNumChannels(const unsigned char* data,const int numPixels)
{
  bool isGray = true;
  bool hasAlpha = false;

  for(int xy=0;xy<numPixels;xy++)
  {
    const unsigned char r = data[xy*4+0];
    const unsigned char g = data[xy*4+1];
    const unsigned char b = data[xy*4+2];
    const unsigned char a = data[xy*4+3];

    if (!(r==g && g==b)) { isGray  = false; }
    if (a<255)           { hasAlpha = true; }
  }

  const int numChannels = (isGray ? 1 : 3) + (hasAlpha ? 1 : 0);

  return numChannels;
}

V2i pyramidLevelSize(const V2i& sizeBase,const int level)
{
  return V2i(V2f(sizeBase)*std::pow(2.0f,-float(level)));
}

std::string backendToString(const int ebsynthBackend)
{
  if      (ebsynthBackend==EBSYNTH_BACKEND_CPU)  { return "cpu";  }
  else if (ebsynthBackend==EBSYNTH_BACKEND_CUDA) { return "cuda"; }
  else if (ebsynthBackend==EBSYNTH_BACKEND_AUTO) { return "auto"; }
  return "unknown";
}

int main(int argc,char** argv)
{
  if (argc<2)
  {
    printf("usage: %s [options]\n",argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -style <style.png>\n");
    printf("  -guide <source.png> <target.png>\n");
    printf("  -output <output.png>\n");
    printf("  -weight <value>\n");
    printf("  -uniformity <value>\n");
    printf("  -patchsize <size>\n");
    printf("  -pyramidlevels <number>\n");
    printf("  -searchvoteiters <number>\n");
    printf("  -patchmatchiters <number>\n");
    printf("  -stopthreshold <value>\n");
    printf("  -extrapass3x3\n");
    printf("  -backend [cpu|cuda]\n");
    printf("\n");
    return 1;
  }

  std::string styleFileName;
  float       styleWeight = -1;
  std::string outputFileName = "output.png";

  struct Guide
  {
    std::string    sourceFileName;
    std::string    targetFileName;
    float          weight;

    int            sourceWidth;
    int            sourceHeight;
    unsigned char* sourceData;

    int            targetWidth;
    int            targetHeight;
    unsigned char* targetData;
    
    int            numChannels;
  };

  std::vector<Guide> guides;

  float uniformityWeight = 3500;
  int patchSize = 5; 
  int numPyramidLevels = -1;
  int numSearchVoteIters = 6;
  int numPatchMatchIters = 4;
  int stopThreshold = 5;
  int extraPass3x3 = 0;
  int backend = ebsynthBackendAvailable(EBSYNTH_BACKEND_CUDA) ? EBSYNTH_BACKEND_CUDA : EBSYNTH_BACKEND_CPU;

  {
    std::vector<std::string> args(argc);
    for(int i=0;i<argc;i++) { args[i] = argv[i]; }
  
    bool fail = false;
    int argi = 1;   

    float* precedingStyleOrGuideWeight = 0;
    while(argi<argc && !fail)
    {
      float weight;
      std::pair<std::string,std::string> guidePair;
      std::string backendName;

      if      (tryToParseStringArg(args,&argi,"-style",&styleFileName,&fail))
      {
        styleWeight = -1;
        precedingStyleOrGuideWeight = &styleWeight;
        argi++;
      }
      else if (tryToParseStringPairArg(args,&argi,"-guide",&guidePair,&fail))
      {
        Guide guide;
        guide.sourceFileName = guidePair.first;
        guide.targetFileName = guidePair.second;
        guide.weight = -1;
        guides.push_back(guide);
        precedingStyleOrGuideWeight = &guides[guides.size()-1].weight;
        argi++;
      }
      else if (tryToParseStringArg(args,&argi,"-output",&outputFileName,&fail))
      {
        argi++;
      }
      else if (tryToParseFloatArg(args,&argi,"-weight",&weight,&fail))
      {
        if (precedingStyleOrGuideWeight!=0)
        {
          if (weight>=0) { *precedingStyleOrGuideWeight = weight; }
          else { printf("error: weights must be non-negaitve!\n"); return 1; }
        }
        else { printf("error: at least one -style or -guide option must precede the -weight option!\n"); return 1; }
        argi++;
      }
      else if (tryToParseFloatArg(args,&argi,"-uniformity",&uniformityWeight,&fail)) { argi++; }
      else if (tryToParseIntArg(args,&argi,"-patchsize",&patchSize,&fail))
      {
        if (patchSize<3)    { printf("error: patchsize is too small!\n"); return 1; }
        if (patchSize%2==0) { printf("error: patchsize must be an odd number!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-pyramidlevels",&numPyramidLevels,&fail))
      {
        if (numPyramidLevels<1) { printf("error: bad argument for -pyramidlevels!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-searchvoteiters",&numSearchVoteIters,&fail))
      {
        if (numSearchVoteIters<0) { printf("error: bad argument for -searchvoteiters!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-patchmatchiters",&numPatchMatchIters,&fail))
      {
        if (numPatchMatchIters<0) { printf("error: bad argument for -patchmatchiters!\n"); return 1; }
        argi++;
      }
      else if (tryToParseIntArg(args,&argi,"-stopthreshold",&stopThreshold,&fail))
      {
        if (stopThreshold<0) { printf("error: bad argument for -stopthreshold!\n"); return 1; }
        argi++;
      }
      else if (tryToParseStringArg(args,&argi,"-backend",&backendName,&fail))
      {
        if      (backendName=="cpu" ) { backend = EBSYNTH_BACKEND_CPU; }
        else if (backendName=="cuda") { backend = EBSYNTH_BACKEND_CUDA; }
        else { printf("error: unrecognized backend '%s'\n",backendName.c_str()); return 1; }

        if (!ebsynthBackendAvailable(backend)) { printf("error: the %s backend is not available!\n",backendToString(backend).c_str()); return 1; }

        argi++;
      }
      else if (argi<args.size() && args[argi]=="-extrapass3x3")
      {
        extraPass3x3 = 1;
        argi++;
      }
      else
      {
        printf("error: unrecognized option '%s'\n",args[argi].c_str());
        fail = true;
      }

    }
    
    if (fail) { return 1; }
  }

  const int numGuides = guides.size();

  int sourceWidth = 0;
  int sourceHeight = 0;
  unsigned char* sourceStyleData = tryLoad(styleFileName,&sourceWidth,&sourceHeight);
  const int numStyleChannelsTotal = evalNumChannels(sourceStyleData,sourceWidth*sourceHeight);

  std::vector<unsigned char> sourceStyle(sourceWidth*sourceHeight*numStyleChannelsTotal);
  for(int xy=0;xy<sourceWidth*sourceHeight;xy++)
  {
    if      (numStyleChannelsTotal>0)  { sourceStyle[xy*numStyleChannelsTotal+0] = sourceStyleData[xy*4+0]; }
    if      (numStyleChannelsTotal==2) { sourceStyle[xy*numStyleChannelsTotal+1] = sourceStyleData[xy*4+3]; }           
    else if (numStyleChannelsTotal>1)  { sourceStyle[xy*numStyleChannelsTotal+1] = sourceStyleData[xy*4+1]; }
    if      (numStyleChannelsTotal>2)  { sourceStyle[xy*numStyleChannelsTotal+2] = sourceStyleData[xy*4+2]; }
    if      (numStyleChannelsTotal>3)  { sourceStyle[xy*numStyleChannelsTotal+3] = sourceStyleData[xy*4+3]; }                 
  }
  
  int targetWidth = 0;
  int targetHeight = 0;
  int numGuideChannelsTotal = 0;

  for(int i=0;i<numGuides;i++)
  {
    Guide& guide = guides[i];

    guide.sourceData = tryLoad(guide.sourceFileName,&guide.sourceWidth,&guide.sourceHeight);
    guide.targetData = tryLoad(guide.targetFileName,&guide.targetWidth,&guide.targetHeight);
      
    if              (guide.sourceWidth!=sourceWidth || guide.sourceHeight!=sourceHeight)  { printf("error: source guide '%s' doesn't match the resolution of '%s'\n",guide.sourceFileName.c_str(),styleFileName.c_str()); return 1; }      
    if      (i>0 && (guide.targetWidth!=targetWidth || guide.targetHeight!=targetHeight)) { printf("error: target guide '%s' doesn't match the resolution of '%s'\n",guide.targetFileName.c_str(),guides[0].targetFileName.c_str()); return 1; }
    else if (i==0) { targetWidth = guide.targetWidth; targetHeight = guide.targetHeight; }

    guide.numChannels = std::max(evalNumChannels(guide.sourceData,sourceWidth*sourceHeight),
                                 evalNumChannels(guide.targetData,targetWidth*targetHeight));    
  
    numGuideChannelsTotal += guide.numChannels;
  }
  
  if (numStyleChannelsTotal>EBSYNTH_MAX_STYLE_CHANNELS) { printf("error: too many style channels (%d), maximum number is %d\n",numStyleChannelsTotal,EBSYNTH_MAX_STYLE_CHANNELS); return 1; }
  if (numGuideChannelsTotal>EBSYNTH_MAX_GUIDE_CHANNELS) { printf("error: too many guide channels (%d), maximum number is %d\n",numGuideChannelsTotal,EBSYNTH_MAX_GUIDE_CHANNELS); return 1; }

  std::vector<unsigned char> sourceGuides(sourceWidth*sourceHeight*numGuideChannelsTotal);
  for(int xy=0;xy<sourceWidth*sourceHeight;xy++)
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  

      if      (numChannels>0)  { sourceGuides[xy*numGuideChannelsTotal+c+0] = guides[i].sourceData[xy*4+0]; }
      if      (numChannels==2) { sourceGuides[xy*numGuideChannelsTotal+c+1] = guides[i].sourceData[xy*4+3]; }           
      else if (numChannels>1)  { sourceGuides[xy*numGuideChannelsTotal+c+1] = guides[i].sourceData[xy*4+1]; }
      if      (numChannels>2)  { sourceGuides[xy*numGuideChannelsTotal+c+2] = guides[i].sourceData[xy*4+2]; }
      if      (numChannels>3)  { sourceGuides[xy*numGuideChannelsTotal+c+3] = guides[i].sourceData[xy*4+3]; }            
      
      c += numChannels;
    }
  }

  std::vector<unsigned char> targetGuides(targetWidth*targetHeight*numGuideChannelsTotal);
  for(int xy=0;xy<targetWidth*targetHeight;xy++)
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  

      if      (numChannels>0)  { targetGuides[xy*numGuideChannelsTotal+c+0] = guides[i].targetData[xy*4+0]; }
      if      (numChannels==2) { targetGuides[xy*numGuideChannelsTotal+c+1] = guides[i].targetData[xy*4+3]; }           
      else if (numChannels>1)  { targetGuides[xy*numGuideChannelsTotal+c+1] = guides[i].targetData[xy*4+1]; }
      if      (numChannels>2)  { targetGuides[xy*numGuideChannelsTotal+c+2] = guides[i].targetData[xy*4+2]; }
      if      (numChannels>3)  { targetGuides[xy*numGuideChannelsTotal+c+3] = guides[i].targetData[xy*4+3]; }            
      
      c += numChannels;
    }
  }

  std::vector<float> styleWeights(numStyleChannelsTotal);
  if (styleWeight<0) { styleWeight = 1.0f; }
  for(int i=0;i<numStyleChannelsTotal;i++) { styleWeights[i] = styleWeight / float(numStyleChannelsTotal); }

  for(int i=0;i<numGuides;i++) { if (guides[i].weight<0) { guides[i].weight = 1.0f/float(numGuides); } }

  std::vector<float> guideWeights(numGuideChannelsTotal);
  {
    int c = 0;
    for(int i=0;i<numGuides;i++)
    { 
      const int numChannels = guides[i].numChannels;  
      
      for(int j=0;j<numChannels;j++)
      {
        guideWeights[c+j] = guides[i].weight / float(numChannels);
      }

      c += numChannels; 
    }
  }

  int maxPyramidLevels = 0;
  for(int level=32;level>=0;level--)
  {
    if (min(pyramidLevelSize(std::min(V2i(sourceWidth,sourceHeight),V2i(targetWidth,targetHeight)),level)) >= (2*patchSize+1))
    {
      maxPyramidLevels = level+1;
      break;
    }
  }

  if (numPyramidLevels==-1) { numPyramidLevels = maxPyramidLevels; }
  numPyramidLevels = std::min(numPyramidLevels,maxPyramidLevels); 

  std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
  std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
  std::vector<int> stopThresholdPerLevel(numPyramidLevels);
  for(int i=0;i<numPyramidLevels;i++)
  {
    numSearchVoteItersPerLevel[i] = numSearchVoteIters;
    numPatchMatchItersPerLevel[i] = numPatchMatchIters;
    stopThresholdPerLevel[i] = stopThreshold;
  }

  std::vector<unsigned char> output(targetWidth*targetHeight*numStyleChannelsTotal);

  printf("uniformity: %.0f\n",uniformityWeight);
  printf("patchsize: %d\n",patchSize);
  printf("pyramidlevels: %d\n",numPyramidLevels);
  printf("searchvoteiters: %d\n",numSearchVoteIters);
  printf("patchmatchiters: %d\n",numPatchMatchIters);
  printf("stopthreshold: %d\n",stopThreshold);
  printf("extrapass3x3: %s\n",extraPass3x3!=0?"yes":"no");
  printf("backend: %s\n",backendToString(backend).c_str());

  ebsynthRun(backend,
             numStyleChannelsTotal,
             numGuideChannelsTotal,
             sourceWidth,
             sourceHeight,
             sourceStyle.data(),
             sourceGuides.data(),
             targetWidth,
             targetHeight,
             targetGuides.data(),
             NULL,
             styleWeights.data(),
             guideWeights.data(),
             uniformityWeight,
             patchSize,
             EBSYNTH_VOTEMODE_PLAIN,
             numPyramidLevels,
             numSearchVoteItersPerLevel.data(),
             numPatchMatchItersPerLevel.data(),
             stopThresholdPerLevel.data(),
             extraPass3x3,
             NULL,
             output.data());

  stbi_write_png(outputFileName.c_str(),targetWidth,targetHeight,numStyleChannelsTotal,output.data(),numStyleChannelsTotal*targetWidth);

  printf("result was written to %s\n",outputFileName.c_str());

  stbi_image_free(sourceStyleData);

  for(int i=0;i<numGuides;i++)
  {
    stbi_image_free(guides[i].sourceData);
    stbi_image_free(guides[i].targetData);
  }
  
  return 0;
}
