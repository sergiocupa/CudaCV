
#ifndef ResizeMultiple3x_H
#define ResizeMultiple3x_H

#ifdef __cplusplus
extern "C" {
#endif


    CUDA_CV_RESULT _Reduce3x(cvBYTE* data, int width, int height, int image_channels, int output_channels, cvBOOL use_single_buffer, cvBOOL result_to_local);


#ifdef __cplusplus
}
#endif

#endif /* ResizeMultiple3x */







