

#include "../Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>



__global__ void Reduce3xGray_Device(CUDA_CV_BUFFER mem, int original_width, int original_height, int resized_width, int resized_height)
{
	const int CHANNELS = 3;
	unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int threadId = (blockIdx.x + threadIdx.x) * THREAD_NUM;

	int stride = original_width * CHANNELS;

	int o_y = pos / resized_width;//         device heigth
	int o_x = pos - (o_y * resized_width);// device width

	int index0 = ((o_y * 3) * stride) + (o_x * 9);// 9 por causa dos canais
	int index1 = index0 + stride;
	int index2 = index1 + stride;

	int result = 0;
	int sum = 0;
	int i = 0;
	while (i < 9)
	{
		sum += mem.Input[index0 + i];
		i++;
	}
	i = 0;
	while (i < 9)
	{
		sum += mem.Input[index1 + i];
		i++;
	}
	i = 0;
	while (i < 9)
	{
		sum += mem.Input[index2 + i];
		i++;
	}
	result = sum / 27;

	mem.Output[pos] = result;
}


__global__ void Reduce3xRGB_Device(CUDA_CV_BUFFER mem, int original_width, int original_height, int resized_width, int resized_height)
{
	const int CHANNELS = 3;
	unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int threadId = (blockIdx.x + threadIdx.x) * THREAD_NUM;

	int stride = original_width * CHANNELS;

	int o_y    = pos / resized_width;//         device heigth
	int o_x    = pos - (o_y * resized_width);// device width
	int index0 = ((o_y * 3) * stride) + (o_x * 3 * CHANNELS);

	double pix[CHANNELS] = { 0,0,0 };
	int i_col;
	int i_color;
	int i_line = 0;
	while (i_line < 3)
	{
		i_col = 0;
		while (i_col < 3)
		{
			i_color = 0;
			while (i_color < CHANNELS)
			{
				pix[i_color] += mem.Input[index0 + i_col + i_color];
				i_color++;
			}
			i_col++;
		}
		i_line++;
		index0 += stride;
	}

	i_color = 0;
	while (i_color < CHANNELS)
	{
		mem.Output[pos + i_color] = pix[i_color] / (3* CHANNELS);
		i_color++;
	}
}



CUDA_CV_RESULT _Reduce3x(cvBYTE* data, int width, int height, int image_channels, int output_channels, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	CUDA_CV_RESULT result;
	result.Status = CUDA_CV_UNKNOWN;
	cudaError_t cudaStatus = cudaSuccess;
	CUDA_CV_GLOBAL global = GetGlobal();

	int resized_width  = width / 3;
	int resized_height = height / 3;
	int stride         = width * image_channels;
	int original_leng  = stride * height;
	int resized_stride = resized_width * output_channels;
	result.Size        = resized_stride * resized_height;

	cudaStatus = cudaMemcpy(global.Reduce3xBuffer.Input, data, original_leng, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		result.Status = CUDA_CV_ERROR_DEVICE_COPY;
		return result;
	}

	dim3 threads = dim3(THREAD_NUM, 1, 1);
	dim3 blocks  = dim3(result.Size / THREAD_NUM, 1, 1);

	if (output_channels == 3)
	{
		Reduce3xRGB_Device << <blocks, threads >> > (global.Reduce3xBuffer, width, height, resized_width, resized_height);
	}
	else
	{
		Reduce3xGray_Device << <blocks, threads >> > (global.Reduce3xBuffer, width, height, resized_width, resized_height);
	}
	//cudaStatus = cudaDeviceSynchronize();

	PrepareResult(global.Reduce3xBuffer, result, result.Size, use_single_buffer, result_to_local);
	return result;
}

