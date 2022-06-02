

#include "../include/CudaCv.cuh"
#include "Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>



void FreeFilterKernels(CUDA_CV_FILTER_KERNEL** fernel, int length, cvBOOL is_device)
{
	if (fernel != NULL)
	{
		if (is_device)
		{
			int ix = 0;
			while (ix < length)
			{
				cudaFree(fernel[ix]);
				ix++;
			}
		}
		else
		{
			int ix = 0;
			while (ix < length)
			{
				free(fernel[ix]);
				ix++;
			}
		}
		free(fernel);
	}
}

CUDA_CV_FILTER_KERNEL** NewFilterKernels(int length, cvBOOL is_device)
{
	CUDA_CV_FILTER_KERNEL** fernel;

	int item_size = sizeof(CUDA_CV_FILTER_KERNEL);
	int cfk_size  = sizeof(CUDA_CV_FILTER_KERNEL*);
	int psize     = length * cfk_size;
	int pos       = 0;

	fernel = malloc(cfk_size);
	if (fernel == NULL) return NULL;

	int fail = 0;
	int ix = 0;

	if (is_device)
	{
		cudaError_t status = cudaSuccess;
		while (ix < length)
		{
			//pos = ix * cfk_size;
			status = cudaMalloc((void**)&fernel[ix], item_size);
			if (status != cudaSuccess)
			{
				fail = 1;
				break;
			}

			cudaMemset(fernel[ix],0, item_size);
			//fernel[ix]->IsDevice = cvTRUE;
			ix++;
		}

		if (fail)
		{
			while (ix > 0)
			{
				ix--;
				cudaFree(fernel[ix]);
			}
			free(fernel);
		}
	}
	else
	{
		while (ix < length)
		{
			//pos = ix * cfk_size;
			fernel[ix] = malloc(item_size);
			
			if (fernel[ix] == NULL)
			{
				fail = 1;
				break;
			}

			memset(fernel[ix], 0, item_size);
			fernel[ix]->IsDevice = cvFALSE;
			ix++;
		}

		if (fail)
		{
			while (ix > 0)
			{
				ix--;
				free(fernel[ix]);
			}
			free(fernel);
		}
	}

	return fernel;
}


void PrepareResult(CUDA_CV_BUFFER buffer, CUDA_CV_RESULT result, int size, cvBOOL use_single_buffer, cvBOOL result_to_local)
{
	cudaError_t cudaStatus = cudaSuccess;

	if (result_to_local)
	{
		result.Data = (cvBYTE*)malloc(size);
		cudaStatus = cudaMemcpy(result.Data, buffer.Output, size, cudaMemcpyDeviceToHost);
	}
	else
	{
		result.IsDevice = cvTRUE;

		if (use_single_buffer)
		{
			result.Data = buffer.Output;
		}
		else
		{
			cudaMalloc((void**)&result.Data, size);
			cudaStatus = cudaMemcpy(result.Data, buffer.Output, size, cudaMemcpyDeviceToDevice);
		}
	}

	if (cudaStatus != cudaSuccess)
	{
		result.Status = CUDA_CV_ERROR_DEVICE_COPY;
	}
}



int _AllocBuffer(CUDA_CV_BUFFER buffer, int size)
{
	cudaError_t status = cudaSuccess;

	if (buffer.IsDevice)
	{
		status = cudaMalloc((void**)&buffer.Input, size);
		if (status != cudaSuccess)
		{
			return status;
		}

		status = cudaMalloc((void**)&buffer.Output, size);
		if (status != cudaSuccess)
		{
			return status;
		}

		if (buffer.UsesInternalBuffer)
		{
			status = cudaMalloc((void**)&buffer.Buffer, size);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
	}
	else
	{
		buffer.Input  = (cvBYTE*)malloc(size);
		buffer.Output = (cvBYTE*)malloc(size);

		if (buffer.UsesInternalBuffer)
		{
			buffer.Buffer = (cvBYTE*)malloc(size);
		}
	}
	return status;
}


void _FreeBuffer(CUDA_CV_BUFFER buffer)
{
	if (buffer.IsDevice)
	{
		if (!buffer.Input)
		{
			cudaFree(buffer.Input);
		}
		if (!buffer.Output)
		{
			cudaFree(buffer.Output);
		}
		if (!buffer.Buffer)
		{
			cudaFree(buffer.Buffer);
		}
	}
	else
	{
		if (!buffer.Input)
		{
			free(buffer.Input);
		}
		if (!buffer.Output)
		{
			free(buffer.Output);
		}
		if (!buffer.Buffer)
		{
			free(buffer.Buffer);
		}
	}
}