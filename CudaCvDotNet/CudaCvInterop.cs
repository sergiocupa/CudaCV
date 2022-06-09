

using System.Runtime.InteropServices;
using System.Security;


namespace CudaCvDotNet
{
    internal class CudaCvInterop
    {

        [SuppressUnmanagedCodeSecurity]
        [DllImport("CudaCv", EntryPoint = "cvReduce3x", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        internal unsafe static extern CUDA_CV_RESULT Reduce3x(byte* data, int width, int height, int image_channels, int output_channels, bool use_single_buffer, bool result_to_local);

        [SuppressUnmanagedCodeSecurity]
        [DllImport("CudaCv", EntryPoint = "cvGaussianBlur", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        internal unsafe static extern CUDA_CV_RESULT GaussianBlur(byte* data, int width, int height, int channels, int gauss_size, bool use_single_buffer, bool result_to_local);





        [SuppressUnmanagedCodeSecurity]
        [DllImport("CudaCv", EntryPoint = "cvInit", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Auto)]
        internal static extern void Init(int max_resotution_x, int max_resolution_y, int image_channels, int max_gaussian_kernel_size);

        [SuppressUnmanagedCodeSecurity]
        [DllImport("CudaCv", EntryPoint = "cvDispose", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Auto)]
        internal static extern void Dispose();

        [SuppressUnmanagedCodeSecurity]
        [DllImport("CudaCv", EntryPoint = "cvSetErrorCallback", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        internal unsafe static extern void _SetErrorCallback(_ImageCudaErrorCallback call);


        internal unsafe delegate void _ImageCudaErrorCallback(int code, int length, byte* description);

    }


    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public unsafe struct CUDA_CV_RESULT
    {
        public CV_STATUS Status;
        public bool      IsDevice;
        public int       Size;
        public byte*     Data;
    }


    public enum CV_STATUS
    {
        CUDA_CV_ERROR_GAUSSIAN_BLUR_INVALID_VALUE_FOR_KERNEL = -4,
        CUDA_CV_ERROR_DEVICE_COPY                            = -3,
        CUDA_CV_ERROR_DEVICE_MALLOC                          = -2,
        CUDA_CV_ERROR_LOCAL_MALLOC                           = -1,
        CUDA_CV_UNKNOWN                                      = 0,
        CUDA_CV_SUCCESS                                      = 1
    }

}
