

#include "Internal.h"
#include "Util.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>


uint8_t* Loopback_Device_Input;
uint8_t* Loopback_Device_Output;



void FreeLoopbackBuffer()
{
	if (Loopback_Device_Input)
	{
		cudaFree(Loopback_Device_Input);
	}

	if (Loopback_Device_Output)
	{
		cudaFree(Loopback_Device_Output);
	}
}



__global__ void Device_LoopbackTest(uint8_t* input, uint8_t* output)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	output[index] = input[index];
}


int TestTransferAndKernel()
{
	cudaError_t status = cudaSuccess;
	status = cudaMalloc((void**)&Loopback_Device_Input, LOOPBACK_TEST_BUFFER_LENGTH);
	if (status != cudaSuccess)
	{
		OnErrorInput(status, "malloc Loopback_Device_Input failed");
		return status;
	}

	status = cudaMalloc((void**)&Loopback_Device_Output, LOOPBACK_TEST_BUFFER_LENGTH);
	if (status != cudaSuccess)
	{
		OnErrorInput(status, "malloc Loopback_Device_Output failed");
		return status;
	}

	uint8_t* ran = CreateRandomArray(LOOPBACK_TEST_BUFFER_LENGTH);
	status = cudaMemcpy(Loopback_Device_Input, ran, LOOPBACK_TEST_BUFFER_LENGTH, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		OnErrorInput(status, "cudaMemcpy Loopback_Device_Input failed");
		return status;
	}

	int nb = LOOPBACK_TEST_BUFFER_LENGTH / THREAD_NUM;
	dim3 threads = dim3(THREAD_NUM, 1, 1);
	dim3 blocks = dim3(nb, 1, 1);
	Device_LoopbackTest << <blocks, threads >> > (Loopback_Device_Input, Loopback_Device_Output);

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		OnErrorInput(status, "cudaMemcpy Loopback_Device_Input failed");
		return status;
	}


	uint8_t* out;
	out = (uint8_t*)malloc(LOOPBACK_TEST_BUFFER_LENGTH);
	status = cudaMemcpy(out, Loopback_Device_Output, LOOPBACK_TEST_BUFFER_LENGTH, cudaMemcpyDeviceToHost);

	FreeLoopbackBuffer();

	int equal = EqualsUInt8(ran, out, LOOPBACK_TEST_BUFFER_LENGTH);
	if (equal != 1)
	{
		OnErrorInput(status, "Loopback retornou dados invalidos");
		return status;
	}

	ResizeMul3_RunTest();

	return status;
}



void _Dispose(CUDA_CV_GLOBAL *global)
{
	if (!global->Initialized) return;

	FreeLoopbackBuffer();
	_FreeBuffer(&global->Reduce3xBuffer);
	_FreeBuffer(&global->GaussianBlurBuffer);

	global->Initialized = 0;
}



void _Init(CUDA_CV_GLOBAL *global, const int max_resolution_x, const int max_resolution_y, const int image_channels, const int max_gaussian_kernel_size)
{
	if (global->Initialized) return;

	global->MaxResolutionX = max_resolution_x;
	global->MaxResolutionY = max_resolution_y;
	global->ImageChannels  = image_channels;
	global->TotalSize      = max_resolution_x * max_resolution_y * image_channels;

	global->Reduce3xBuffer.IsDevice               = cvTRUE;
	global->Reduce3xBuffer.UsesInternalBuffer     = cvFALSE;
	global->GaussianBlurBuffer.IsDevice           = cvTRUE;
	global->GaussianBlurBuffer.UsesInternalBuffer = cvTRUE;

	global->LastError = cudaSetDevice(0);
	if (global->LastError != cudaSuccess)
	{
		_Dispose(global);
		OnErrorInput(global->LastError, "cudaSetDevice failed");
		return;
	}

	global->LastError = (cudaError_t)TestTransferAndKernel();
	if (global->LastError != cudaSuccess)
	{
		_Dispose(global);
		return;
	}

	global->LastError = (cudaError_t)_AllocBuffer(&global->Reduce3xBuffer, global->TotalSize);
	if (global->LastError != cudaSuccess)
	{
		_Dispose(global);
		OnErrorInput(global->LastError, "AllocBuffer Reduce3xBuffer failed");
		return;
	}

	global->LastError = (cudaError_t)_AllocBuffer(&global->GaussianBlurBuffer, global->TotalSize);
	if (global->LastError != cudaSuccess)
	{
		_Dispose(global);
		OnErrorInput(global->LastError, "AllocBuffer GaussianBlurBuffer failed");
		return;
	}

    if (global->LastError) return;

	CreateGaussianKernels(global->BlurFilterKernels, max_gaussian_kernel_size, 1.0);

	global->Initialized = 1;

	//RunMatrixTest();
}