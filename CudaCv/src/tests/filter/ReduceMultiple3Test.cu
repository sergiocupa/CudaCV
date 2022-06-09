
#include "../../Internal.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>


const int TESTE_SIZE = 108;

cvBYTE MatrixTest[TESTE_SIZE] =
{
	100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,
	118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,
	136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,
	154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,
	172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,
	190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207
};

const int RESIZED_LENG = 2 * 6;

cvBYTE Resized[RESIZED_LENG];
cvBYTE ResizedResult[RESIZED_LENG] = { 121,122,123,130,131,132,175,176,177,184,185,186 };

cvBYTE* DeviceInput;
cvBYTE* DeviceOutput;

__global__ void MatrixTestKernel(cvBYTE* Input, cvBYTE* Output, int width, int stride, int resized_stride, int channels, int div, int col_length)
{
	//unsigned int pos = ((blockIdx.x * div) * blockDim.x) + (threadIdx.x * channels);
	unsigned int pos = ((blockIdx.x * channels) * blockDim.x) + (threadIdx.x * channels);

	int ay   = pos / resized_stride;
	int ax   = (pos - (ay * resized_stride));
	int rep  = (ay * div * stride);
	int ix   = rep + (ax * div);

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
				ch[ix_ch] += Input[ix + col + ix_ch];
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
		Output[pos + ix_ch] = ch[ix_ch] / col_length;
		ix_ch++;
	}
}


int ResizeMul3_RunTest()
{
	cudaError_t status = cudaSuccess;
	int leng = sizeof(cvBYTE) * TESTE_SIZE;
	//int WIDTH = 6;

	memset(&Resized, 0, 2*6);

	status = cudaMalloc((void**)&DeviceInput, leng);
	status = cudaMalloc((void**)&DeviceOutput, leng);
	status = cudaMemcpy(DeviceInput, &MatrixTest, leng, cudaMemcpyHostToDevice);

	dim3 threads_per_block = dim3(2, 1, 1);
	dim3 blocks            = dim3(2, 1, 1);

	int CHANNELS = 3;
	int DIV = 3;
	int COL_LENG = DIV * CHANNELS;
	int width = 6;
	int height = 6;
	int resized_w = width / DIV;
	int resizd_h = height / DIV;
	int stride = width * CHANNELS;
	int resized_stride = resized_w * CHANNELS;

	MatrixTestKernel<<<blocks,threads_per_block>>>(DeviceInput, DeviceOutput, width, stride, resized_stride, CHANNELS, DIV, COL_LENG);

	status = cudaGetLastError();
	status = cudaDeviceSynchronize();
	status = cudaMemcpy(&Resized, DeviceOutput, leng, cudaMemcpyDeviceToHost);
	status = cudaFree((void**)&DeviceInput);
	status = cudaFree((void**)&DeviceOutput);

	int success = 1;
	int rix = 0;
	while (rix < RESIZED_LENG)
	{
		if (Resized[rix] != ResizedResult[rix])
		{
			success = 0;
			break;
		}
		rix++;
	}
	return success;
}



    //int b_ix = 0, t_ix = 0;
	//while (b_ix < 2)
	//{
	//	t_ix = 0;
	//	while (t_ix < 2)
	//	{
	//		int pos = ((b_ix * CHANNELS) * 2) + (t_ix * CHANNELS);// 2 = DIM
	//		int ay  = pos / resized_stride;
	//		int ax  = (pos - (ay * resized_stride));
	//		int rep = (ay * DIV * stride);
	//		int ix  = rep + (ax * DIV);

	//		int ch[3] = {0,0,0};
	//		int ix_ch = 0;
	//		int col = 0;
	//		int line = 0;
	//		while (line < DIV)
	//		{
	//			col = 0;
	//			while (col < COL_LENG)
	//			{
	//				ix_ch = 0;
	//				while (ix_ch < CHANNELS)
	//				{
	//					ch[ix_ch] += MatrixTest[ix + col + ix_ch];
	//			        ix_ch++;
	//				}
	//				col += CHANNELS;
	//			}
	//			ix += stride;
	//			line++;
	//		}

	//		ix_ch = 0;
	//		while (ix_ch < CHANNELS)
	//		{
	//			Resized[pos + ix_ch] = ch[ix_ch] / COL_LENG;
	//	        ix_ch++;
	//		}

	//		t_ix++;
	//	}
	//	b_ix++;
	//}

