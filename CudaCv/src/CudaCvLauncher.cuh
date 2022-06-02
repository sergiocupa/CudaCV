
#ifndef CudaCvLauncher_H
#define CudaCvLauncher_H

#ifdef __cplusplus
extern "C" {
#endif


    void _Init(CUDA_CV_GLOBAL global, const int max_resolution_x, const int max_resolution_y, const int image_channels, const int max_gaussian_kernel_size);
    void _Dispose(CUDA_CV_GLOBAL global);


#ifdef __cplusplus
}
#endif

#endif /* CudaCvLauncher */