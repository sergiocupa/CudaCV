

#include "../Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>


static double* GetGaussianKernel(int size);



__global__ void convoluteGaussianBlur_Device(CUDA_CV_BUFFER mem, double* bernel, int width, int stride, int channels, int k_off)
{
	unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;
	/*int y = pos / width;
	int x = pos - (y * width);*/
	int y       = pos / stride;
    int x       = pos - (y * stride);
	int kcenter = (y * stride) + (x * channels);

	int kpixel  = 0;
	int posc = 0;
	int c = 0;
	while (c < channels)
	{
		mem.Buffer[pos + c] = 0;
		c++;
	}
	
	int fx;
	int fy = -k_off;
	while (fy <= k_off)
	{
		fx = -k_off;
		while (fx <= k_off)
		{
			kpixel = kcenter + (fy * stride) + (fx * channels);
			c = 0;
			while (c < channels)
			{
				mem.Buffer[pos + c] += (double)mem.Input[kpixel + c] * bernel[fy + k_off, fx + k_off];
				c++;
			}
			fx++;
		}
		fy++;
	}

	c = 0;
	while (c < channels)
	{
		posc = pos + c;
		if (mem.Buffer[posc] > 255)
		{
			mem.Buffer[posc] = 255;
		}
		else if (mem.Buffer[posc] < 0)
		{
			mem.Buffer[posc] = 0;
		}
		c++;
	}

	c = 0;
	while (c < channels)
	{
		posc = pos + c;
		mem.Output[posc] = (cvBYTE)mem.Buffer[posc];
		c++;
	}
}



CUDA_CV_RESULT convoluteGaussianBlur(cvBYTE *image_data, int width, int height, int channels, int gauss_size, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	CUDA_CV_GLOBAL global = GetGlobal();
	cudaError_t cudaStatus = cudaSuccess;
	CUDA_CV_RESULT result;
	double* gkernel = GetGaussianKernel(gauss_size);

	int k_off    = (gauss_size - 1) / 2;
	int stride   = width * channels;
	int length   = stride * height;
	int rgb_leng = channels * sizeof(double);

	cudaStatus = cudaMemcpy(global.GaussianBlurBuffer.Input, image_data, length, cudaMemcpyHostToDevice);

	dim3 threads = dim3(THREAD_NUM, 1, 1);
	dim3 blocks  = dim3(length / THREAD_NUM, 1, 1);
	convoluteGaussianBlur_Device<<<blocks,threads>>>
    (
		global.GaussianBlurBuffer,
		gkernel,
		width,
		stride,
		channels,
		k_off
    );

	PrepareResult(global.GaussianBlurBuffer, result, result.Size, use_single_buffer, result_to_local);
	return result;
}




static cvBYTE* convoluteGaussianBlur_Local(cvBYTE* data, int width, int height, int channels, int gauss_size, double* kernel)
{
	cvBYTE* result;
	int stride = width * channels;
	int length = stride * height;

	double* rgb;
	rgb = (double*)malloc(channels * sizeof(double));
	result = (cvBYTE*)malloc(length);

	int k_off = (gauss_size - 1) / 2;
	int kcenter = 0;
	int kpixel = 0;

	for (int y = k_off; y < height - k_off; y++)
	{
		for (int x = k_off; x < width - k_off; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				rgb[c] = 0.0;
			}
			kcenter = (y * stride) + (x * channels);

			for (int fy = -k_off; fy <= k_off; fy++)
			{
				for (int fx = -k_off; fx <= k_off; fx++)
				{
					kpixel = kcenter + (fy * stride) + (fx * channels);
					for (int c = 0; c < channels; c++)
					{
						rgb[c] += (double)(data[kpixel + c]) * kernel[fy + k_off, fx + k_off];
					}
				}
			}

			for (int c = 0; c < channels; c++)
			{
				if (rgb[c] > 255)
				{
					rgb[c] = 255;
				}
				else if (rgb[c] < 0)
				{
					rgb[c] = 0;
				}
			}

			for (int c = 0; c < channels; c++)
			{
				result[kcenter + c] = (uint8_t)rgb[c];
			}
		}
	}

	return result;
}


cvBYTE* convoluteGaussianBlurCPU(cvBYTE* data, int width, int height, int channels, int gauss_size)
{
	cvBYTE* result;
	double* gkernel = GetGaussianKernel(gauss_size);
	result = convoluteGaussianBlur_Local(data, width, height, channels, gauss_size, gkernel);
	return result;
}



static double* GetGaussianKernel(int size)
{
	CUDA_CV_GLOBAL global = GetGlobal();

	if (global.FilterKernels.DeviceKernels == 0) return 0;

	int i = 0;
	while (i <= global.FilterKernels.Length)
	{
		if (global.FilterKernels.DeviceKernels[i]->Width == size)
		{
			return global.FilterKernels.DeviceKernels[i]->Kernel;
		}
		i++;
	}
	return NULL;
}