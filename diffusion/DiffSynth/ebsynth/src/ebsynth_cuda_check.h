#ifndef EBSYNTH_CUDA_CHECK_H_
#define EBSYNTH_CUDA_CHECK_H_

template<typename T>
bool checkCudaError_(T result,char const* const func,const char* const file,int const line)
{
  if (result)
  {
    printf("CUDA error at %s:%d code=%d \"%s\"\n",file,line,static_cast<unsigned int>(result),func);
    return true;
  }
  else
  {
    return false;
  }
}

#define checkCudaError(val) checkCudaError_((val),#val,__FILE__,__LINE__)

#endif
