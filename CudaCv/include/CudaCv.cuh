#ifndef CudaCv_H
#define CudaCv_H

#ifdef __cplusplus
extern "C" {
#endif


    #ifdef WIN32 || WIN64
        #define CUDA_CV_WIN
    #endif 


    #ifdef CUDA_CV_WIN
        #define CUDA_CV_API __declspec ( dllexport )
    #else 
        #define CUDA_CV_API
    #endif 


    #include <stdint.h>

    #define cvBYTE uint8_t
    #define cvBOOL int
    #define cvFALSE 0
    #define cvTRUE 1



    typedef void (*ImageCudaErrorCallback)(const int code, const int length, const char* description);



    typedef enum _CUDA_CV_STATUS
    {
        CUDA_CV_ERROR_GAUSSIAN_BLUR_INVALID_VALUE_FOR_KERNEL = -4,
        CUDA_CV_ERROR_DEVICE_COPY                            = -3,
        CUDA_CV_ERROR_DEVICE_MALLOC                          = -2,
        CUDA_CV_ERROR_LOCAL_MALLOC                           = -1,
        CUDA_CV_UNKNOWN                                      = 0,
        CUDA_CV_SUCCESS                                      = 1
    }
    CUDA_CV_STATUS;


    typedef struct _COLOR_3CH_8BITS
    {
        cvBYTE A;
        cvBYTE B;
        cvBYTE C;
    }
    COLOR_3CH_8BITS;


    typedef struct _CUDA_CV_RESULT
    {
        CUDA_CV_STATUS Status;
        cvBOOL         IsDevice;
        int            Size;
        cvBYTE*        Data;
    }
    CUDA_CV_RESULT;


    typedef struct _CUDA_CV_IMAGE
    {
        cvBYTE* Data;
        int Width;
        int Height;
        int Channels;
        int Dept;
    }
    CUDA_CV_IMAGE;



    CUDA_CV_API
    void cvInit(const int max_resotution_x, const int max_resolution_y, const int image_channels, const int max_gaussian_kernel_size);

    
    CUDA_CV_API
    void cvDispose();


    CUDA_CV_API
    void cvSetErrorCallback(ImageCudaErrorCallback call);

    
    CUDA_CV_API
    CUDA_CV_RESULT cvReduce3x(cvBYTE* data, int width, int height, int image_channels, int output_channels, cvBOOL use_single_buffer, cvBOOL result_to_local);


    CUDA_CV_RESULT cvGaussianBlur(cvBYTE* image_data, int width, int height, int channels, int gauss_size, cvBOOL use_single_buffer, cvBOOL result_to_local);



#ifdef __cplusplus
}
#endif

#endif /* CudaCv */