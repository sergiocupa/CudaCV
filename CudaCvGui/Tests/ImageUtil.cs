

using System.Windows.Media;
using System.Windows.Media.Imaging;


namespace CudaCvGui.Tests
{
    public class ImageUtil
    {

        public static BitmapSource ToBitmapBGR24(byte[] data, int width, int height)
        {
            var bgr    = PixelFormats.Bgr24;
            var stride = (width * bgr.BitsPerPixel + 7) / 8;
            var result = BitmapSource.Create(width, height, 96, 96, bgr, null, data, stride);
            return result;
        }

        public static BitmapSource ToBitmapGray8(byte[] data, int width, int height)
        {
            var gray    = PixelFormats.Gray8;
            var result = BitmapSource.Create(width, height, 96, 96, gray, null, data, width);
            return result;
        }



    }
}
