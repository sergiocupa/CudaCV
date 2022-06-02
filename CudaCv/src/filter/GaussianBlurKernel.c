

#include "../Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>



static void CreateGaussianKernel(double* kr, const int size, double alpha)
{
	int pos;
	int    foff = (size - 1) / 2;
	double r, s = foff * alpha * alpha;
	double rrr;

	double sum = 0.0;

	for (int x = -foff; x <= foff; x++)
	{
		for (int y = -foff; y <= foff; y++)
		{
			r = sqrt(x * x + y * y);
			pos = ((y + foff) * size) + (x + foff);
			*(kr + pos) = (exp(-(r * r) / s)) / (M_PI * s);
			sum += *(kr + pos);
		}
	}

	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			pos = (j * size) + i;
			*(kr + pos) /= sum;
		}
	}
	rrr = 0;
}


CUDA_CV_STATUS CreateGaussianKernels(CUDA_CV_FILTER_KERNELS kernels, const int count, const double alpha)
{
	const int MIN_SIZE = 3;

	if(count < MIN_SIZE) return CUDA_CV_ERROR_GAUSSIAN_BLUR_INVALID_VALUE_FOR_KERNEL;
	int mod = count % 2;
	if (mod != 1) return CUDA_CV_ERROR_GAUSSIAN_BLUR_INVALID_VALUE_FOR_KERNEL;

	int leng = (count -1) / 2.0;
	leng--;

	int cfk_size = sizeof(CUDA_CV_FILTER_KERNEL*);
	int psize    = leng * cfk_size;

	kernels.Length        = leng;
	kernels.LocalKernels  = NewFilterKernels(leng, cvFALSE);// malloc(psize);
	kernels.DeviceKernels = NewFilterKernels(leng, cvTRUE);// malloc(psize);

	if (kernels.LocalKernels == NULL || kernels.DeviceKernels == NULL)
	{
		OnErrorInput(CUDA_CV_ERROR_LOCAL_MALLOC, "local malloc LocalKernels/DeviceKernels % i failed", count);
		return CUDA_CV_ERROR_LOCAL_MALLOC;
	}

	cudaError_t status = cudaSuccess;
	kernels.Length = 0;

	int size = 0;
	int refs = MIN_SIZE;
	int pos = 0;
	int ix = 0;
	while (ix < leng)
	{
		size = (refs * refs) * sizeof(double);
		pos  = ix * cfk_size;

		kernels.LocalKernels[pos]->Kernel = (double*)malloc(size);
		cudaMalloc((void**)&kernels.DeviceKernels[pos]->Kernel, size);

		kernels.LocalKernels[pos]->Width   = refs;
		kernels.LocalKernels[pos]->Height  = refs;
		kernels.DeviceKernels[pos]->Kernel = refs;
		kernels.DeviceKernels[pos]->Width  = refs;
		kernels.DeviceKernels[pos]->Height = refs;

		status = cudaMalloc((void**)&(*kernels.DeviceKernels)[pos].Kernel, size);
		if (status != cudaSuccess) 
		{
			OnErrorInput(status, "device malloc CreateGaussianKernels %i failed", refs);
			return CUDA_CV_ERROR_DEVICE_MALLOC;
		}

		CreateGaussianKernel(kernels.LocalKernels[pos]->Kernel, count, alpha);

		status = cudaMemcpy(kernels.DeviceKernels[pos]->Kernel,(double*)&kernels.LocalKernels[pos]->Kernel,size,cudaMemcpyHostToDevice);
		if (status != cudaSuccess) 
		{ 
			OnErrorInput(status, "device memcopy CreateGaussianKernels %i failed", refs);
			return; 
		}

		ix++;
		refs += 2;
	}

	kernels.Length = ix;
	return CUDA_CV_UNKNOWN;
}