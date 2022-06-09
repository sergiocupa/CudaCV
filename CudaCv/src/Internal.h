#ifndef INTERNAL_H
#define INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../include/CudaCv.cuh"
    #include "Util.h"
    #include <stdint.h>

    #define BLOCK_DIM                   32
    #define THREAD_NUM                  1024
    #define WARP_SIZE                   32
    #define LOOPBACK_TEST_BUFFER_LENGTH 99328


    typedef enum _FILTER_KERNEL_TYPE
    {
        FILTER_KERNEL_UNKNOWN       = 0,
        FILTER_KERNEL_GAUSSIAN_BLUR = 1,
    }
    FILTER_KERNEL_TYPE;


    typedef struct _CUDA_CV_FILTER_KERNEL
    {
        cvBOOL IsDevice;
        int Size;
        int Width;
        int Height;
        double *Kernel;
    }
    CUDA_CV_FILTER_KERNEL;


    typedef struct _CUDA_CV_FILTER_KERNELS
    {
        int Length;
        FILTER_KERNEL_TYPE Type;
        CUDA_CV_FILTER_KERNEL **LocalKernels;
        CUDA_CV_FILTER_KERNEL **DeviceKernels;
    }
    CUDA_CV_FILTER_KERNELS;


    typedef struct _CUDA_CV_BUFFER
    {
        cvBOOL  UsesInternalBuffer;
        cvBOOL  IsDevice;
        cvBYTE *Input;
        cvBYTE *Output;
        cvBYTE *Buffer;
    }
    CUDA_CV_BUFFER;

   
    typedef struct _CUDA_CV_GLOBAL
    {
        int Initialized;
        int LastError;
        int MaxResolutionX;
        int MaxResolutionY;
        int ImageChannels;
        int TotalSize;
        CUDA_CV_BUFFER  GaussianBlurBuffer;
        CUDA_CV_BUFFER  Reduce3xBuffer;
        CUDA_CV_FILTER_KERNELS BlurFilterKernels;
    }
    CUDA_CV_GLOBAL;


    void FreeFilterKernels(CUDA_CV_FILTER_KERNEL** fernel, int length);
    CUDA_CV_FILTER_KERNEL** NewFilterKernels(int length);


    void PrepareResult(CUDA_CV_BUFFER buffer, CUDA_CV_RESULT *result, int size, cvBOOL use_single_buffer, cvBOOL result_to_local);
    void _FreeBuffer(CUDA_CV_BUFFER *buffer);
    int _AllocBuffer(CUDA_CV_BUFFER *buffer, int size);
    CUDA_CV_GLOBAL GetGlobal();


    // Tests
    int ResizeMul3_RunTest();


    #include "CudaCvLauncher.cuh"
    #include "filter/GaussianBlurKernel.h"
    #include "filter/ImageConvolution.cuh"
    #include "filter/ResizeMultiple3x.cuh"


#ifdef __cplusplus
}
#endif

#endif /* INTERNAL */