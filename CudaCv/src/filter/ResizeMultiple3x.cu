

#include "../Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



//__global__ void Reduce3xGray_Device(CUDA_CV_BUFFER mem, int original_width, int resized_width)
//{
//	const int CHANNELS = 3;
//	unsigned int pos = threadIdx.x + blockIdx.x * blockDim.x;
//	//unsigned int threadId = (blockIdx.x + threadIdx.x) * THREAD_NUM;
//
//	int stride = original_width * CHANNELS;
//
//	int o_y = pos / resized_width;//         device heigth
//	int o_x = pos - (o_y * resized_width);// device width
//
//	int index0 = ((o_y * 3) * stride) + (o_x * 9);// 9 por causa dos canais
//	int index1 = index0 + stride;
//	int index2 = index1 + stride;
//
//	int result = 0;
//	int sum = 0;
//	int i = 0;
//	/*while (i < 9)
//	{
//		sum += mem.Input[index0 + i];
//		i++;
//	}
//	i = 0;
//	while (i < 9)
//	{
//		sum += mem.Input[index1 + i];
//		i++;
//	}
//	i = 0;
//	while (i < 9)
//	{
//		sum += mem.Input[index2 + i];
//		i++;
//	}
//	result = sum / 27;
//
//	mem.Output[pos] = result;*/
//}


__global__ void Reduce3xRGB_Device(CUDA_CV_BUFFER mem, int width, int stride, int resized_stride, int channels, int div, int col_length)
{
	unsigned int pos = ((blockIdx.x * channels) * blockDim.x) + (threadIdx.x * channels);

	int ay  = pos / resized_stride;
	int ax  = (pos - (ay * resized_stride));
	int rep = (ay * div * stride);
	int ix  = rep + (ax * div);

	int ch[3] = { 0,0,0 };
	int ix_ch = 0;
	int col;
	int line = 0;
	while (line < div)
	{
		col = 0;
		while (col < col_length)
		{
			ix_ch = 0;
			while (ix_ch < channels)
			{
				ch[ix_ch] += mem.Input[ix + col + ix_ch];
				ix_ch++;
			}
			col += channels;
		}
		ix += stride;
		line++;
	}

	ix_ch = 0;
	while (ix_ch < channels)
	{
		mem.Output[pos + ix_ch] = ch[ix_ch] / col_length;
		ix_ch++;
	}
}



CUDA_CV_RESULT _Reduce3x(cvBYTE* data, int width, int height, int image_channels, int output_channels, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	CUDA_CV_RESULT result;
	result.Status   = CUDA_CV_UNKNOWN;
	result.IsDevice = cvFALSE;
	cudaError_t cudaStatus = cudaSuccess;
	CUDA_CV_GLOBAL global = GetGlobal();

	int CHANNELS       = 3;
	int DIV            = 3;
	int COL_LENG       = DIV * CHANNELS;
	int resized_width  = width / DIV;
	int resized_height = height / DIV;
	int stride         = width * CHANNELS;
	int resized_stride = resized_width * CHANNELS;
	int original_leng  = height * width * CHANNELS * sizeof(cvBYTE);
	int mat_size       = resized_width * resized_height;
	result.Size        = mat_size * output_channels;

	cudaStatus = cudaMemcpy(global.Reduce3xBuffer.Input, data, original_leng, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		result.Status = CUDA_CV_ERROR_DEVICE_COPY;
		return result;
	}

	dim3 threads_per_block = dim3(THREAD_NUM, 1, 1);
	dim3 blocks            = dim3(mat_size / THREAD_NUM, 1, 1);

	if (output_channels == 3)
	{
		Reduce3xRGB_Device<<<blocks, threads_per_block >>>(global.Reduce3xBuffer, width, stride, resized_stride, CHANNELS, DIV, COL_LENG);
	}
	else
	{
		//Reduce3xGray_Device<<<blocks, threads_per_block >>>(global.Reduce3xBuffer, width, resized_width);
	}
	//cudaStatus = cudaDeviceSynchronize();

	PrepareResult(global.Reduce3xBuffer, &result, result.Size, use_single_buffer, result_to_local);
	return result;
}

