
#pragma once

#define WIN32_LEAN_AND_MEAN  
#include <windows.h>

#include "include/CudaCv.cuh"


LONG WINAPI MyUnhandledExceptionFilter(PEXCEPTION_POINTERS pExceptionPtrs)
{
    // Do something, for example generate error report
    //..
    // Execute default exception handler next
    return EXCEPTION_EXECUTE_HANDLER;
}


BOOL APIENTRY DllMain( HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        {
            SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
        }
        break;
        case DLL_THREAD_ATTACH:
        {
            int a = 1;
        }
        break;
        case DLL_THREAD_DETACH:
        {
            int b = 1;
        }
        break;
        case DLL_PROCESS_DETACH:
        {
            int c = 1;
        }
        break;
    }
    return TRUE;
}

