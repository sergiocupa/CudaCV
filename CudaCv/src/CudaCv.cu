

#include "Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>


CUDA_CV_GLOBAL Global;
CUDA_CV_GLOBAL GetGlobal() { return Global; }



void cvInit(const int max_resotution_x, const int max_resolution_y, const int image_channels, const int max_gaussian_kernel_size)
{
    _Init(Global, max_resotution_x, max_resolution_y, image_channels, max_gaussian_kernel_size);
}


void cvDispose()
{
	_Dispose(Global);
}


void cvSetErrorCallback(ImageCudaErrorCallback call)
{
	SetErrorCallback(call);
}


CUDA_CV_RESULT cvReduce3x(cvBYTE* data, int width, int height, int image_channels, int output_channels, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	return _Reduce3x(data, width, height, image_channels, output_channels, use_single_buffer, result_to_local);
}

CUDA_CV_RESULT cvGaussianBlur(cvBYTE* image_data, int width, int height, int channels, int gauss_size, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	return convoluteGaussianBlur(image_data, width, height, channels, gauss_size, use_single_buffer, result_to_local);
}




