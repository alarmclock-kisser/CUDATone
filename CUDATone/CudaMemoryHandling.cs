
using ManagedCuda;
using ManagedCuda.BasicTypes;
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

		public List<IntPtr> PushChunks<T>(List<T[]> data, bool silent = false) where T : unmanaged
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

			// Create list of pointers
			List<IntPtr> pointers = new();
			
			// Push each chunk
			foreach (T[] chunk in data)
			{
				IntPtr pointer = this.PushData(chunk, silent);
				pointers.Add(pointer);
			}

			return pointers;
		}

		public List<T[]> PullChunks<T>(List<IntPtr> pointers, bool free = false, bool silent = false) where T : unmanaged
		{
			// Create list of chunks
			List<T[]> chunks = new();
			
			// Pull each chunk
			foreach (IntPtr pointer in pointers)
			{
				T[] chunk = this.PullData<T>(pointer, free, silent);
				chunks.Add(chunk);
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
			// Get total memory
			long totalSize = this.Context.GetTotalDeviceMemorySize();
			
			// Convert to megabytes
			if (asMegabytes)
			{
				totalSize /= 1024 * 1024;
			}

			return totalSize;
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
			// Get total memory usage
			long totalSize = this.GetTotalMemoryUsage(true, true);

			// Get total memory available
			long totalAvailable = this.GetTotalMemory(true);

			// Update progress bar
			this.VramBar.Maximum = (int) totalAvailable;
			this.VramBar.Value = (int) totalSize;
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