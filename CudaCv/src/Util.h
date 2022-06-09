


#ifndef Util_H
#define Util_H

#ifdef __cplusplus
extern "C" {
#endif

    #include "../include/CudaCv.cuh"
    #include <stdint.h>


	int EqualsUInt8(uint8_t* a1, uint8_t* a2, int length);

	void DebugPrint(const char* description_format, ...);

	void OnErrorInput(int code, const char* description_format, ...);

    void SetErrorCallback(ImageCudaErrorCallback call);

	uint8_t* CreateRandomArray(const int length);

	long util_get_nanosecond(void);
	long util_get_microsecond(void);
    long util_get_milisecond(void);

	int IsDebugMode();


#ifdef __cplusplus
}
#endif

#endif /* Util */