using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace CudaCvGui.Tests
{
    internal class StaRunner
    {

        internal static void Run(Action act)
        {
            ManualResetEvent mre = new ManualResetEvent(false);

            Action inner = () =>
            {
                act();
                mre.Set();
            };

            var thr = new Thread(new ThreadStart(inner));
            thr.SetApartmentState(ApartmentState.STA);
            thr.Start();
            mre.Reset();
            mre.WaitOne();
        }
    }
}
