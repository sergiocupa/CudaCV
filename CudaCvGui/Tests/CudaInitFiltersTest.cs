

using CudaCvDotNet;
using Microsoft.Win32;
using System;
using System.IO;
using System.Threading;
using System.Windows;


namespace CudaCvGui.Tests
{


 

    internal class CudaInitFiltersTest
    {


        internal static void Start(Window window)
        {
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
                    var gau = CudaCV.GaussianBlur(bytes, Width, Height, 3, 21);

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
