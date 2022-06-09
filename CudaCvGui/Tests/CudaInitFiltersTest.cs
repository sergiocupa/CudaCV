

using CudaCvDotNet;
using Microsoft.Win32;
using System;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Linq;


namespace CudaCvGui.Tests
{


 

    internal class CudaInitFiltersTest
    {

        static MainWindow Window;


        internal static void Start(MainWindow window)
        {
            Window = window;

            try
            {
                int Width  = 1920;
                int Height = 1080;

                string file = "";

                StaRunner.Run(() => 
                {
                    var diag = new OpenFileDialog();
                    diag.ShowDialog();
                    file = diag.FileName;
                });

                if (!string.IsNullOrEmpty(file))
                {
                    var bytes = File.ReadAllBytes(file);

                    CudaCV.Init(2000, 2000, 3, 31);

                    var red = CudaCV.Reduce3x(bytes, Width, Height, 3, 3);
                    var aaa = ImageUtil.ToBitmapBGR24(red, Width / 3, Height / 3);
                    Window.OriginalImage.Source = aaa;

                    CudaCV.Dispose();
                }
               
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }


    }
}
