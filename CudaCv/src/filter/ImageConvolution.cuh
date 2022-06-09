#ifndef ImageConvolution_H
#define ImageConvolution_H

#ifdef __cplusplus
extern "C" {
#endif


    CUDA_CV_RESULT convoluteGaussianBlur(cvBYTE* image_data, int width, int height, int channels, int gauss_size, cvBOOL use_single_buffer, cvBOOL result_to_local);
    cvBYTE* convoluteGaussianBlurCPU(cvBYTE* data, int width, int height, int channels, int gauss_size);


#ifdef __cplusplus
}
#endif

#endif /* ImageConvolution */