
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace CUDATone
{
	public class CudaMemoryHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private PrimaryContext Context;
		private ProgressBar VramBar;

		public List<CudaBuffer> Buffers = [];


		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaMemoryHandling(string repopath, ListBox logList, PrimaryContext context, ProgressBar? vramBar = null)
		{
			this.Repopath = repopath;
			this.LogList = logList;
			this.Context = context;
			this.VramBar = vramBar ?? new ProgressBar();
		}





		// ----- ----- METHODS ----- ----- \\
		// Log
		public void Log(string message = "", string inner = "", int indent = 0)
		{
			string msg = $"[Memory]: {new string(' ', indent * 2)}{message}{(string.IsNullOrEmpty(inner) ? "" : $" ({inner})")}";

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


		/// Dispose & free
		public void Dispose()
		{
			// Free buffers
		}

		public long FreeBuffer(IntPtr pointer, bool readable = false)
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return 0;
			}

			// Set current
			this.Context.SetCurrent();

			// Get size
			long size = this.GetBufferSize(pointer, readable);

			// Get device ptr
			CUdeviceptr ptr = new CUdeviceptr(pointer);

			// Free buffer
			this.Context.FreeMemory(ptr);

			// Remove from dict
			this.Buffers.Remove(obj);

			// Update progress bar
			this.UpdateProgressBar();

			return size;
		}


		/// Get buffer + info
		public CudaBuffer? GetBuffer(IntPtr pointer)
		{
			// Find buffer obj by pointer
			CudaBuffer? obj = this.Buffers.FirstOrDefault(x => x.Pointer == pointer);
			if (obj == null)
			{
				// Log
				this.Log($"Couldn't find", "<" + pointer + ">", 1);
				
				return null;
			}

			// Update progress bar
			this.UpdateProgressBar();

			return obj;
		}

		public Type GetBufferType(IntPtr pointer)
		{
			Type defaultType = typeof(void);

			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return defaultType;
			}

			return obj.Type;
		}

		public long GetBufferSize(IntPtr pointer, bool readable = false)
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null)
			{
				return 0;
			}

			// Get buffer type
			Type bufferType = obj.Type;

			// Get length in bytes
			long length = (long) obj.Length * Marshal.SizeOf(bufferType);

			// Make readable
			if (readable)
			{
				length /= 1024 * 1024;
			}

			return length;
		}


		// Push & pull
		public IntPtr PushData<T>(IEnumerable<T> data, bool silent = false) where T : unmanaged
		{
			// Check data
			if (data == null || !data.Any())
			{
				if (!silent)
				{
					this.Log("No data to push", "", 1);
				}
				return 0;
			}

			// Set current
			this.Context.SetCurrent();

			// Get length pointer
			IntPtr length = (nint) data.LongCount();

			// Allocate buffer & copy data
			CudaDeviceVariable<T> buffer = new(length);
			buffer.CopyToDevice(data.ToArray());

			// Get pointer
			IntPtr pointer = buffer.DevicePointer.Pointer;

			// Log
			if (!silent)
			{
				this.Log($"Pushed {length * Marshal.SizeOf<T>() / 1024} kB", "<" + pointer + ">", 1);
			}

			// Create obj
			CudaBuffer obj = new()
			{
				Pointer = pointer,
				Length = length,
				Type = typeof(T)
			};

			// Add to dict
			this.Buffers.Add(obj);

			// Update progress bar
			this.UpdateProgressBar();

			// Return pointer
			return pointer;
		}

		public T[] PullData<T>(IntPtr pointer, bool free = false, bool silent = false) where T : unmanaged
		{
			// Get buffer
			CudaBuffer? obj = this.GetBuffer(pointer);
			if (obj == null || obj.Length == 0)
			{
				return [];
			}

			// Set current
			this.Context.SetCurrent();

			// Create array with long count
			T[] data = new T[obj.Length];

			// Get device pointer
			CUdeviceptr ptr = new(pointer);

			// Copy data to host from device pointer
			this.Context.CopyToHost(data, ptr);

			// Log
			if (!silent)
			{
				this.Log($"Pulled {obj.Size / 1024} kB", "<" + pointer + ">", 1);
			}

			// Free buffer
			if (free)
			{
				this.FreeBuffer(pointer);
			}

			// Update progress bar
			this.UpdateProgressBar();

			// Return data
			return data;
		}

		public IntPtr[] PushChunks<T>(List<T[]> data, bool silent = false) where T : unmanaged
		{
			// Check data
			if (data == null || !data.Any())
			{
				if (!silent)
				{
					this.Log("No data to push", "", 1);
				}
				return [];
			}

			// Create list of pointers + stopwatch
			List<IntPtr> pointers = [];
			Stopwatch sw = Stopwatch.StartNew();

			// Push each chunk
			foreach (T[] chunk in data)
			{
				IntPtr pointer = this.PushData(chunk, true);
				pointers.Add(pointer);
			}

			// Log
			sw.Stop();
			if (!silent)
			{
				long sizeKb = pointers.Sum(x => this.GetBufferSize(x)) / 1024;
				this.Log($"Pushed {pointers.Count} chunks", $"{sizeKb} kb, {sw.ElapsedMilliseconds} ms", 1);
			}

			return pointers.ToArray();
		}

		public List<T[]> PullChunks<T>(IntPtr[] pointers, bool free = false, bool silent = false) where T : unmanaged
		{
			// Create list of chunks + stopwatch
			List<T[]> chunks = [];
			Stopwatch sw = Stopwatch.StartNew();
			
			// Pull each chunk
			foreach (IntPtr pointer in pointers)
			{
				T[] chunk = this.PullData<T>(pointer, free, true);
				chunks.Add(chunk);
			}

			// Log
			sw.Stop();
			if (!silent)
			{
				long sizeKb = chunks.Sum(x => (long) x.Length * Marshal.SizeOf(typeof(T))) / 1024;
				this.Log($"Pulled {chunks.Count} chunks", $"{sizeKb} kB, {sw.ElapsedMilliseconds} ms", 1);
			}

			return chunks;
		}


		/// Allocate buffer
		public IntPtr AllocateBuffer<T>(IntPtr length, bool silent = false) where T : unmanaged
		{
			// Check length
			if (length < 1)
			{
				if (!silent)
				{
					this.Log("No length to allocate", "", 1);
				}

				return 0;
			}

			this.Context.SetCurrent();

			// Allocate buffer
			CudaDeviceVariable<T> buffer = new(length);
			
			// Get pointer
			IntPtr pointer = buffer.DevicePointer.Pointer;
			
			// Create obj
			CudaBuffer obj = new()
			{
				Pointer = pointer,
				Length = length,
				Type = typeof(T)
			};

			// Log
			if (!silent)
			{
				this.Log($"Allocated {length / 1024} kB", "<" + pointer + ">", 1);
			}

			// Add to dict
			this.Buffers.Add(obj);

			// Update PBar
			this.UpdateProgressBar();

			return pointer;
		}


		// Get memory info
		public long GetTotalMemoryUsage(bool actual = false, bool asMegabytes = false)
		{
			// Sum up all buffer sizes * sizeof(type)
			long totalSize = this.Buffers.Sum(x => (long) x.Length * Marshal.SizeOf(x.Type));

			// Get total memory
			long totalAvailable = this.GetTotalMemory() - this.Context.GetFreeDeviceMemorySize();
			if (actual)
			{
				totalSize = totalAvailable;
			}

			// Convert to megabytes
			if (asMegabytes)
			{
				totalSize /= 1024 * 1024;
			}

			return totalSize;
		}

		public long GetTotalMemory(bool asMegabytes = false)
		{
			try
			{
				// Ensure context is current
				this.Context.SetCurrent();

				// Get total memory
				long totalSize = this.Context.GetTotalDeviceMemorySize();

				// Convert to megabytes
				if (asMegabytes)
				{
					totalSize /= 1024 * 1024;
				}

				return totalSize;
			}
			catch (CudaException ex)
			{
				this.Log("CUDA Context error: " + ex.Message, "GetTotalMemory()", 1);
				throw;
			}
		}


		public long GetFreeMemory(bool asMegabytes = false)
		{
			// Get free memory
			long totalSize = this.Context.GetFreeDeviceMemorySize();
			
			// Convert to megabytes
			if (asMegabytes)
			{
				totalSize /= 1024 * 1024;
			}

			return totalSize;
		}


		// UI
		public void UpdateProgressBar()
		{
			try
			{
				// Holen, falls im falschen Thread
				if (this.VramBar.InvokeRequired)
				{
					this.VramBar.Invoke(new Action(this.UpdateProgressBar));
					return;
				}

				// Aktueller Thread ist der UI-Thread: sicher fortfahren
				long totalSize = this.GetTotalMemoryUsage(true, true);
				long totalAvailable = this.GetTotalMemory(true);

				// Clamp falls zu groß
				int safeValue = (int) Math.Min(totalSize, this.VramBar.Maximum);
				int safeMax = (int) Math.Max(safeValue, 1); // Minimum 1

				this.VramBar.Maximum = (int) totalAvailable;
				this.VramBar.Value = safeValue;
			}
			catch (Exception ex)
			{
				this.Log("Error refreshin progress bar value: " + ex.Message, "UpdateProgressBar", 1);
			}
		}

	}



	public class CudaBuffer
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public IntPtr Pointer { get; set; }
		public IntPtr Length { get; set; }
		public Type Type { get; set; } = typeof(void);

		public long Size => this.Length * Marshal.SizeOf(this.Type);





	}
}