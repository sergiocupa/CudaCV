#ifndef GaussianBlurKernel_H
#define GaussianBlurKernel_H

#ifdef __cplusplus
extern "C" {
#endif

    //#include "../Internal.h"


    CUDA_CV_STATUS CreateGaussianKernels(CUDA_CV_FILTER_KERNELS kernels, const int count, const double alpha);


#ifdef __cplusplus
}
#endif

#endif // GaussianBlurKernel 