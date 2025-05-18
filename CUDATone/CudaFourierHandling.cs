using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDATone
{
	public class CudaFourierHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private PrimaryContext Context;

		private CudaMemoryHandling MemoryH;


		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaFourierHandling(string repopath, ListBox logList, PrimaryContext context, CudaMemoryHandling memoryH)
		{
			this.Repopath = repopath;
			this.LogList = logList;
			this.Context = context;
			this.MemoryH = memoryH;
		}





		// ----- ----- METHODS ----- ----- \\
		public void Log(string message = "", string inner = "", int indent = 0)
		{
			string msg = $"[Fourier]: {new string(' ', indent * 2)}{message}{(string.IsNullOrEmpty(inner) ? "" : $" ({inner})")}";

			if (this.LogList.InvokeRequired)
			{
				this.LogList.Invoke((MethodInvoker) (() => {
					this.LogList.Items.Add(msg);
					this.LogList.SelectedIndex = this.LogList.Items.Count - 1;
				}));
			}
			else
			{
				this.LogList.Items.Add(msg);
				this.LogList.SelectedIndex = this.LogList.Items.Count - 1;
			}
		}


		public void Dispose()
		{
			// Free plans etc.
		}



		public IntPtr[] ExecuteFFT(IntPtr[] pointers, bool keep = false, bool silent = false)
		{
			// Result list
			List<IntPtr> result = [];

			// Get length
			IntPtr length = this.MemoryH.GetBuffer(pointers.FirstOrDefault())?.Length ?? 0;


			// Create plan
			CudaFFTPlan1D plan = new((int)length, cufftType.R2C, 1);

			// Loop through pointers
			foreach (IntPtr pointer in pointers)
			{
				// Get buffer
				CudaBuffer? obj = this.MemoryH.GetBuffer(pointer);
				if (obj == null || obj.Length == 0)
				{
					if (!silent)
					{
						this.Log($"Couldn't get buffer", "<" + pointer + ">", 1);
					}
					continue;
				}

				// Create output buffer
				IntPtr complexPointer = this.MemoryH.AllocateBuffer<float2>(length, silent);
				CudaBuffer? complexBuffer = this.MemoryH.GetBuffer(complexPointer);
				if (complexBuffer == null)
				{
					if (!silent)
					{
						// Log
						this.Log($"Couldn't allocate complex", "<" + complexPointer + ">", 1);
					}

					return pointers;
				}
				result.Add(complexPointer);

				// Get CUdevice pointer
				CUdeviceptr devicePointer = new(pointer);
				CUdeviceptr complexDevicePointer = new(complexPointer);

				// Exec
				plan.Exec(devicePointer, complexDevicePointer);

				if (!silent)
				{
					// Log
					this.Log($"Executed FFT", "<" + pointer + ">", 2);
				}

				// Free buffer if not keeping
				if (!keep)
				{
					this.MemoryH.FreeBuffer(pointer, silent);
					if (!silent)
					{
						this.Log($"Freed buffer", "<" + pointer + ">", 2);
					}
				}
			}

			// Free plan
			plan.Dispose();

			if (!silent)
			{
				this.Log($"Freed plan", $"{result.Count} transformed", 1);
			}

			// Return result
			return result.ToArray();
		}

		public IntPtr[] ExecuteIFFT(IntPtr[] pointers, bool keep = false, bool silent = false)
		{
			// Result list
			List<IntPtr> result = [];

			// Get length
			IntPtr length = this.MemoryH.GetBuffer(pointers.FirstOrDefault())?.Length ?? 0;


			// Create plan
			CudaFFTPlan1D plan = new((int) length, cufftType.C2R, 1);

			// Loop through pointers
			foreach (IntPtr pointer in pointers)
			{
				// Get buffer
				CudaBuffer? obj = this.MemoryH.GetBuffer(pointer);
				if (obj == null || obj.Length == 0)
				{
					if (!silent)
					{
						this.Log($"Couldn't get buffer", "<" + pointer + ">", 1);
					}
					continue;
				}

				// Create output buffer
				IntPtr floatPointer = this.MemoryH.AllocateBuffer<float>(length, silent);
				CudaBuffer? floatBuffer = this.MemoryH.GetBuffer(floatPointer);
				if (floatBuffer == null)
				{
					if (!silent)
					{
						// Log
						this.Log($"Couldn't allocate float", "<" + floatPointer + ">", 1);
					}

					return pointers;
				}
				result.Add(floatPointer);

				// Get CUdevice pointer
				CUdeviceptr devicePointer = new(pointer);
				CUdeviceptr floatDevicePointer = new(floatPointer);

				// Exec
				plan.Exec(devicePointer, floatDevicePointer);

				if (!silent)
				{
					// Log
					this.Log($"Executed IFFT", "<" + pointer + ">", 2);
				}

				// Free buffer if not keeping
				if (!keep)
				{
					this.MemoryH.FreeBuffer(pointer);
					if (!silent)
					{
						this.Log($"Freed buffer", "<" + pointer + ">", 2);
					}
				}
			}

			// Free plan
			plan.Dispose();

			if (!silent)
			{
				this.Log($"Freed plan", $"{result.Count} transformed", 1);
			}

			// Return result
			return result.ToArray();
		}



		public async Task<IntPtr[]> ExecuteFFTAsync(IntPtr[] pointers, bool keep = false, bool silent = false, ProgressBar? pBar = null)
		{
			return await Task.Run(() =>
			{
				List<IntPtr> result = [];

				if (pointers == null || pointers.Length == 0)
				{
					return result.ToArray();
				}

				IntPtr length = this.MemoryH.GetBuffer(pointers.FirstOrDefault())?.Length ?? 0;
				if (length == IntPtr.Zero)
				{
					return result.ToArray();
				}

				var plan = new CudaFFTPlan1D((int) length, cufftType.R2C, 1);

				int total = pointers.Length;
				int processed = 0;

				foreach (IntPtr pointer in pointers)
				{
					var obj = this.MemoryH.GetBuffer(pointer);
					if (obj == null || obj.Length == 0)
					{
						if (!silent)
						{
							this.Log($"Couldn't get buffer", "<" + pointer + ">", 1);
						}

						continue;
					}

					IntPtr complexPointer = this.MemoryH.AllocateBuffer<float2>(length, silent);
					var complexBuffer = this.MemoryH.GetBuffer(complexPointer);
					if (complexBuffer == null)
					{
						if (!silent)
						{
							this.Log($"Couldn't allocate complex", "<" + complexPointer + ">", 1);
						}

						return pointers;
					}

					result.Add(complexPointer);

					var devicePointer = new CUdeviceptr(pointer);
					var complexDevicePointer = new CUdeviceptr(complexPointer);

					plan.Exec(devicePointer, complexDevicePointer);

					if (!silent)
					{
						this.Log($"Executed FFT", "<" + pointer + ">", 2);
					}

					if (!keep)
					{
						this.MemoryH.FreeBuffer(pointer, silent);
						if (!silent)
						{
							this.Log($"Freed buffer", "<" + pointer + ">", 2);
						}
					}

					// ProgressBar update
					processed++;
					if (pBar != null && pBar.InvokeRequired)
					{
						pBar.Invoke(() => pBar.Value = Math.Min(pBar.Maximum, processed * pBar.Maximum / total));
					}
				}

				plan.Dispose();

				if (!silent)
				{
					this.Log($"Freed plan", $"{result.Count} transformed", 1);
				}

				return result.ToArray();
			});
		}


		public async Task<IntPtr[]> ExecuteIFFTAsync(IntPtr[] pointers, bool keep = false, bool silent = false, ProgressBar? pBar = null)
		{
			return await Task.Run(() =>
			{
				List<IntPtr> result = [];

				if (pointers == null || pointers.Length == 0)
				{
					return result.ToArray();
				}

				IntPtr length = this.MemoryH.GetBuffer(pointers.FirstOrDefault())?.Length ?? 0;
				if (length == IntPtr.Zero)
				{
					return result.ToArray();
				}

				var plan = new CudaFFTPlan1D((int) length, cufftType.C2R, 1);

				int total = pointers.Length;
				int processed = 0;

				foreach (IntPtr pointer in pointers)
				{
					var obj = this.MemoryH.GetBuffer(pointer);
					if (obj == null || obj.Length == 0)
					{
						if (!silent)
						{
							this.Log($"Couldn't get buffer", "<" + pointer + ">", 1);
						}

						continue;
					}

					IntPtr floatPointer = this.MemoryH.AllocateBuffer<float>(length, silent);
					var floatBuffer = this.MemoryH.GetBuffer(floatPointer);
					if (floatBuffer == null)
					{
						if (!silent)
						{
							this.Log($"Couldn't allocate float", "<" + floatPointer + ">", 1);
						}

						return pointers;
					}

					result.Add(floatPointer);

					var devicePointer = new CUdeviceptr(pointer);
					var floatDevicePointer = new CUdeviceptr(floatPointer);

					plan.Exec(devicePointer, floatDevicePointer);

					if (!silent)
					{
						this.Log($"Executed IFFT", "<" + pointer + ">", 2);
					}

					if (!keep)
					{
						this.MemoryH.FreeBuffer(pointer);
						if (!silent)
						{
							this.Log($"Freed buffer", "<" + pointer + ">", 2);
						}
					}

					// ProgressBar update
					processed++;
					if (pBar != null && pBar.InvokeRequired)
					{
						pBar.Invoke(() => pBar.Value = Math.Min(pBar.Maximum, processed * pBar.Maximum / total));
					}
				}

				plan.Dispose();

				if (!silent)
				{
					this.Log($"Freed plan", $"{result.Count} transformed", 1);
				}

				return result.ToArray();
			});
		}




	}
}
