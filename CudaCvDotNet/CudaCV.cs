

using System.Runtime.InteropServices;
using System.Text;


namespace CudaCvDotNet
{
    public class CudaCV
    {


        public static byte[] GaussianBlur(byte[] data, int width, int height, int channels, int gauss_size)
        {
            CV_STATUS status = CV_STATUS.CUDA_CV_UNKNOWN;
            byte[] result = new byte[0];
            unsafe
            {
                CUDA_CV_RESULT re;
                fixed (byte* p = data, r = result)
                {
                    re = CudaCvInterop.GaussianBlur(p, width, height, channels, gauss_size, false, true);
                    status = re.Status;
                    if (status == CV_STATUS.CUDA_CV_UNKNOWN || status == CV_STATUS.CUDA_CV_SUCCESS)
                    {
                        result = new byte[re.Size];
                        Marshal.Copy(new IntPtr(re.Data), result, 0, re.Size);
                    }
                }
            }

            if (!(status == CV_STATUS.CUDA_CV_UNKNOWN || status == CV_STATUS.CUDA_CV_SUCCESS))
            {
                throw new Exception("CudaCV.GaussianBlur()\r\n" + status.ToString());
            }
            return result;
        }


        public static byte[] Reduce3x(byte[] data, int width, int height, int image_channels, int output_channels)
        {
            CV_STATUS status = CV_STATUS.CUDA_CV_UNKNOWN;
            byte[] result = new byte[0];
            unsafe
            {
                CUDA_CV_RESULT re;
                fixed (byte* p = data, r = result)
                {
                    re = CudaCvInterop.Reduce3x(p, width, height, image_channels, output_channels, false, true);
                    status = re.Status;
                    if (status == CV_STATUS.CUDA_CV_UNKNOWN || status == CV_STATUS.CUDA_CV_SUCCESS)
                    {
                        result = new byte[re.Size];
                        Marshal.Copy(new IntPtr(re.Data), result, 0, re.Size);
                    }
                }
            }

            if (!(status == CV_STATUS.CUDA_CV_UNKNOWN || status == CV_STATUS.CUDA_CV_SUCCESS))
            {
                throw new Exception("CudaCV.Reduce3x()\r\n" + status.ToString());
            }
            return result;
        }





        public static void SetErrorCallback(ImageCudaErrorCallback call)
        {
            unsafe
            {
                CudaCvInterop._SetErrorCallback(FErrorCallback);
            }
            ErrorCallback = call;
        }

        private unsafe static void FErrorCallback(int code, int length, byte* description)
        {
            int rcode = 0;
            string result = "";
            unsafe
            {
                result = Encoding.ASCII.GetString(description, length);
                rcode = code;
            }

            if (ErrorCallback != null)
            {
                ErrorCallback(rcode, result);
            }
        }

        private static ImageCudaErrorCallback ErrorCallback;


        public static void Init(int max_resotution_x, int max_resolution_y, int image_channels, int max_gaussian_kernel_size)
        {
            CudaCvInterop.Init(max_resotution_x, max_resolution_y, image_channels, max_gaussian_kernel_size);
        }

        public static void Dispose()
        {
            CudaCvInterop.Dispose();
        }


    }


    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public struct CvResult
    {
        byte[] Output;
        CV_STATUS Status;
    }



    public delegate void ImageCudaErrorCallback(int code, string description);
}
