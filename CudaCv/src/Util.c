
#include "Util.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <windows.h>
#include <string.h>


#define UTIL_MESSAGE_SIZE 4096

static long ntime;
ImageCudaErrorCallback CallError;



int EqualsUInt8(uint8_t* a1, uint8_t* a2, int length)
{
	if (!a1 || !a2) return -1;

	int i = 0;
	while (i < length)
	{
		if (a1[i] != a2[i])
		{
			return 0;
		}
		i++;
	}
	return 1;
}


uint8_t* CreateRandomArray(const int length)
{
	uint8_t* result = NULL;
	result = (uint8_t*)malloc(length);
	srand((unsigned int)time(0) + getpid());

	int i = 0;
	while (i < length)
	{
		result[i] = rand() % 255;
		i++;
	}
	return result;
}


void DebugPrint(const char* description_format, ...)
{
	char description[UTIL_MESSAGE_SIZE];
	int leng = strlen(description_format);
	int bkeng = sizeof(description_format);

	va_list vl;

	va_start(vl, description_format);
	vsnprintf(description, bkeng, description_format, vl);
	va_end(vl);

	description[leng] = '\0';
	OutputDebugStringA(description);
}


void OnErrorInput(int code, const char* description_format, ...)
{
	char description[UTIL_MESSAGE_SIZE];
	int leng  = strlen(description_format);
	int bkeng = sizeof(description_format);

	va_list vl;

	va_start(vl, description_format);
	vsnprintf(description, bkeng, description_format, vl);
	va_end(vl);

	description[leng] = '\0';

	if (CallError) CallError(code, leng, description_format);
}


void SetErrorCallback(ImageCudaErrorCallback call)
{
	CallError = call;
}




static long util_get_nanos(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

long util_get_nanosecond(void)
{
	long current = util_get_nanos();
	long result = current - ntime;
	ntime = current;
	return result;
}

long util_get_microsecond(void)
{
	long current = util_get_nanos();
	long result = current - ntime;
	ntime = current;
	return result / 1000;
}

long util_get_milisecond(void)
{
	long current = util_get_nanos();
	long result = current - ntime;
	ntime = current;
	return result / 1000000;
}


int IsDebugMode()
{
	#ifdef _DEBUG
		return 1;
	#else 
		return 0;
	#endif 
}