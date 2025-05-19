
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace CUDATone
{
	public class CudaKernelHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private PrimaryContext Context;
		private CudaMemoryHandling MemoryH;
		private ComboBox KernelsCombo;

		public CudaKernel? Kernel = null;
		public string? KernelName = null;
		public string? KernelFile = null;
		public string? KernelCode = null;


		public List<string> SourceFiles => this.GetCuFiles();
		public List<string> CompiledFiles => this.GetPtxFiles();


		private string KernelPath => Path.Combine(this.Repopath, "Resources", "Kernels");

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaKernelHandling(string repopath, ListBox logList, PrimaryContext context, CudaMemoryHandling memoryH, ComboBox kernelsCombo)
		{
			this.Repopath = repopath;
			this.LogList = logList;
			this.Context = context;
			this.MemoryH = memoryH;
			this.KernelsCombo = kernelsCombo;

			// Register events
			// this.KernelsCombo.SelectedIndexChanged += (s, e) => this.LoadKernel(this.KernelsCombo.SelectedItem?.ToString() ?? "");

			// Compile all kernels
			this.CompileAll(true, true);

			// Fill kernels combobox
			this.FillKernelsCombo();
		}




		// ----- ----- METHODS ----- ----- \\
		// Log
		public void Log(string message = "", string inner = "", int indent = 0)
		{
			string msg = $"[Kernel]: {new string('~', indent)}{message}{(string.IsNullOrEmpty(inner) ? "" : $" ({inner})")}";

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


		// Dispose
		public void Dispose()
		{
			// Dispose of kernels
			this.UnloadKernel();
		}



		// I/O
		public List<string> GetPtxFiles(string? path = null)
		{
			path ??= Path.Combine(this.KernelPath, "PTX");

			// Get all PTX files in kernel path
			string[] files = Directory.GetFiles(path, "*.ptx").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}

		public List<string> GetCuFiles(string? path = null)
		{
			path ??= Path.Combine(this.KernelPath, "CU");

			// Get all CU files in kernel path
			string[] files = Directory.GetFiles(path, "*.cu").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}



		// UI
		public void FillKernelsCombo(int select = -1)
		{
			this.KernelsCombo.Items.Clear();

			// Get all PTX files in kernel path
			string[] files = this.CompiledFiles.Select(f => Path.GetFileNameWithoutExtension(f)).ToArray();

			// Add to combo box
			foreach (string file in files)
			{
				this.KernelsCombo.Items.Add(file);
			}

			// Select first item
			if (this.KernelsCombo.Items.Count > select)
			{
				this.KernelsCombo.SelectedIndex = select;
			}
		}



		// (Un) Load kernel
		public void SelectLatestKernel()
		{
			string[] files = this.CompiledFiles.ToArray();

			// Get file info (last modified), sort by last modified date, select latest
			FileInfo[] fileInfos = files.Select(f => new FileInfo(f)).OrderByDescending(f => f.LastWriteTime).ToArray();
			
			string latestFile = fileInfos.FirstOrDefault()?.FullName ?? "";
			string latestName = Path.GetFileNameWithoutExtension(latestFile) ?? "";
			this.KernelsCombo.SelectedItem = latestName;
		}

		public CudaKernel? LoadKernel(string kernelName, bool silent = false)
		{
			if (this.Context == null)
			{
				this.Log("No CUDA context available", "", 1);
				return null;
			}

			// Unload?
			if (this.Kernel != null)
			{
				this.UnloadKernel();
			}

			// Get kernel path
			string kernelPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");

			// Get log path
			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			// Log
			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				this.Log("Started loading kernel " + kernelName);
			}

			// Try to load kernel
			try
			{
				// Load ptx code
				byte[] ptxCode = File.ReadAllBytes(kernelPath);

				string cuPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");

				// Load kernel
				this.Kernel = this.Context.LoadKernelPTX(ptxCode, kernelName);
				this.KernelName = kernelName;
				this.KernelFile = kernelPath;
				this.KernelCode = File.ReadAllText(cuPath);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					this.Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				this.Kernel = null;
			}

			// Log
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				this.Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			return this.Kernel;
		}

		public void UnloadKernel()
		{
			// Unload kernel
			if (this.Kernel != null)
			{
				this.Context.UnloadKernel(this.Kernel);
				this.Kernel = null;
			}
		}



		// Compile kernel
		public string? CompileKernel(string filepath, bool silent = false)
		{
			if (this.Context == null)
			{
				if (!silent)
				{
					this.Log("No CUDA available", "", 1);
				}
				return null;
			}

			// If file is not a .cu file, but raw kernel string, compile that
			if (Path.GetExtension(filepath) != ".cu")
			{
				return this.CompileString(filepath, silent);
			}

			string kernelName = Path.GetFileNameWithoutExtension(filepath);

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				this.Log("Compiling kernel '" + kernelName + "'");
			}

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);
				string log = rtc.GetLogAsString();

				if (log.Length > 0)
				{
					// Count double \n
					int count = log.Split(["\n\n"], StringSplitOptions.None).Length - 1;
					if (!silent)
					{
						this.Log("Compiled with warnings", count.ToString(), 1);
					}
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}

				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				if (!silent)
				{
					this.Log("Compiled within " + deltaMicros + " µs", "Repo\\" + Path.GetRelativePath(this.Repopath, logpath), 1);
				}

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				if (!silent)
				{
					this.Log("PTX exported", ptxPath, 1);
				}

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				this.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return null;
			}

		}

		public string? CompileString(string kernelString, bool silent = false)
		{
			if (this.Context == null)
			{
				if (!silent)
				{
					this.Log("No CUDA available", "", 1);
				}
				return null;
			}

			string kernelName = kernelString.Split("void ")[1].Split("(")[0];

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				this.Log("Compiling kernel '" + kernelName + "'");
			}

			// Load kernel file
			string kernelCode = kernelString;

			// Save also the kernel string as .c file
			string cPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");
			File.WriteAllText(cPath, kernelCode);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);
				string log = rtc.GetLogAsString();

				if (log.Length > 0)
				{
					// Count double \n
					int count = log.Split(["\n\n"], StringSplitOptions.None).Length - 1;
					if (!silent)
					{
						this.Log("Compiled with warnings", count.ToString(), 1);
					}
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				if (!silent)
				{
					this.Log("Compiled within " + deltaMicros + " µs", "Repo\\" + Path.GetRelativePath(this.Repopath, logpath), 1);
				}


				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				if (!silent)
				{
					this.Log("PTX exported", ptxPath, 1);
				}

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				this.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return null;
			}
		}

		public string? PrecompileKernelString(string kernelString, bool silent = false)
		{
			// Check contains "extern c"
			if (!kernelString.Contains("extern \"C\""))
			{
				if (!silent)
				{
					this.Log("Kernel string does not contain 'extern \"C\"'", "", 1);
				}
				return null;
			}

			// Check contains "__global__ "
			if (!kernelString.Contains("__global__"))
			{
				if (!silent)
				{
					this.Log("Kernel string does not contain '__global__'", "", 1);
				}
				return null;
			}

			// Check contains "void "
			if (!kernelString.Contains("void "))
			{
				if (!silent)
				{
					this.Log("Kernel string does not contain 'void '", "", 1);
				}
				return null;
			}

			// Check contains int
			if (!kernelString.Contains("int ") && !kernelString.Contains("long "))
			{
				if (!silent)
				{
					this.Log("Kernel string does not contain 'int ' (for array length)", "", 1);
				}
				return null;
			}

			// Check if every bracket is closed (even amount) for {} and () and []
			int open = kernelString.Count(c => c == '{');
			int close = kernelString.Count(c => c == '}');
			if (open != close)
			{
				if (!silent)
				{
					this.Log("Kernel string has unbalanced brackets", " { } ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '(');
			close = kernelString.Count(c => c == ')');
			if (open != close)
			{
				if (!silent)
				{
					this.Log("Kernel string has unbalanced brackets", " ( ) ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '[');
			close = kernelString.Count(c => c == ']');
			if (open != close)
			{
				if (!silent)
				{
					this.Log("Kernel string has unbalanced brackets", " [ ] ", 1);
				}
				return null;
			}

			// Check if kernel contains "blockIdx.x" and "blockDim.x" and "threadIdx.x"
			if (!kernelString.Contains("blockIdx.x") || !kernelString.Contains("blockDim.x") || !kernelString.Contains("threadIdx.x"))
			{
				if (!silent)
				{
					this.Log("Kernel string should contain 'blockIdx.x', 'blockDim.x' and 'threadIdx.x'", "", 2);
				}
			}

			// Get name between "void " and "("
			int start = kernelString.IndexOf("void ") + "void ".Length;
			int end = kernelString.IndexOf("(", start);
			string name = kernelString.Substring(start, end - start);

			// Trim every line ends from empty spaces (split -> trim -> aggregate)
			kernelString = kernelString.Split("\n").Select(x => x.TrimEnd()).Aggregate((x, y) => x + "\n" + y);

			// Log name
			if (!silent)
			{
				this.Log("Succesfully precompiled kernel string", "Name: " + name, 1);
			}

			return name;
		}

		public void CompileAll(bool silent = false, bool logErrors = false)
		{
			List<string> sourceFiles = this.SourceFiles;

			// Compile all source files
			foreach (string sourceFile in sourceFiles)
			{
				string? ptx = this.CompileKernel(sourceFile, silent);
				if (string.IsNullOrEmpty(ptx) && logErrors)
				{
					this.Log("Compilation failed: ", Path.GetFileNameWithoutExtension(sourceFile), 1);
				}
			}
		}




		// Execute kernel (audio)
		public IntPtr ExecuteKernelAudio(IntPtr pointer, IntPtr inputLength, object[] arguments, IntPtr expectedResultLength = 0, Type? expectedResultType = null, int channels = 2, int bitdepth = 32, bool silent = false)
		{
			// 1. Kernel-Check
			if (this.Kernel == null)
			{
				if (!silent)
				{
					this.Log("Kernel not loaded", this.KernelName ?? "N/A", 1);
				}

				return pointer;
			}

			// 2. Argumente prüfen
			Dictionary<string, Type> args = this.GetArguments(null, silent);

			// 3. Puffer-Infos holen
			CUdeviceptr devicePtr = new(pointer);
			expectedResultType ??= this.MemoryH.GetBufferType(pointer);
			if (expectedResultLength < 1)
			{
				expectedResultLength = this.MemoryH.GetBufferSize(pointer, true);
			}

			// 4. Ausgabe-Puffer allokieren (falls nötig)
			IntPtr outputPointer = 0;
			if (args.Count(x => x.Value == typeof(IntPtr)) == 2)
			{
				// Generische Allokation basierend auf expectedResultType
				dynamic buffer = this.MemoryH.AllocateBuffer(expectedResultType, expectedResultLength, silent);
				outputPointer = ((CUdeviceptr) buffer).Pointer;
			}

			// 5. Kernel-Argumente mergen
			CUdeviceptr outputPtr = new(outputPointer);
			object[] kernelArgs = this.MergeArguments(devicePtr, inputLength, outputPtr,
													expectedResultLength, channels, bitdepth,
													arguments, silent);

			// 6. 1D-Dimensionen für Audio-Daten
			long totalSamples = (long) inputLength / (bitdepth / 8); // Anzahl der Float-Samples
			int blockSize = 256; // Typische Blockgröße für 1D-Kernel
			int gridSize = (int) ((totalSamples + blockSize - 1) / blockSize);

			this.Kernel.BlockDimensions = new dim3(blockSize, 1, 1);  // 1D-Block
			this.Kernel.GridDimensions = new dim3(gridSize, 1, 1);    // 1D-Grid

			// 7. Kernel ausführen
			this.Kernel.Run(kernelArgs);

			if (!silent)
			{
				this.Log("Kernel executed", this.KernelName ?? "N/A", 1);
			}

			// 8. Eingabe-Puffer freigeben (falls Ausgabe-Puffer existiert)
			if (outputPointer != 0)
			{
				this.MemoryH.FreeBuffer(devicePtr.Pointer);
			}

			// 9. Synchronisieren
			this.Context.Synchronize();

			return outputPointer != 0 ? outputPointer : pointer;
		}

		public async Task<IntPtr[]> ExecuteKernelAudioBatchParallelAsync(IntPtr[] pointers, IntPtr inputLength, object[] arguments, IntPtr expectedResultLength = 0, Type? expectedResultType = null, int channels = 2, int bitdepth = 32, ProgressBar? pBar = null, bool silent = false)
		{
			// 1. Kernel-Check
			if (this.Kernel == null)
			{
				if (!silent)
				{
					this.Log("Kernel not loaded", this.KernelName ?? "N/A", 1);
				}

				return pointers;
			}

			// 2. Validate inputs
			if (pointers == null || pointers.Length == 0)
			{
				return [];
			}

			// 3. Get common parameters
			Dictionary<string, Type> args = this.GetArguments(null, silent);
			expectedResultType ??= this.MemoryH.GetBufferType(pointers[0]);
			if (expectedResultLength < 1)
			{
				expectedResultLength = this.MemoryH.GetBufferSize(pointers[0], true);
			}

			// 4. Determine if we need output buffers
			bool needsOutput = args.Count(x => x.Value == typeof(IntPtr)) == 2;
			nint[] results = new IntPtr[pointers.Length];

			// 5. Process chunks in parallel
			int processed = 0;
			object progressLock = new();

			await Task.Run(() =>
			{
				Parallel.For(0, pointers.Length, i =>
				{
					try
					{
						// 5a. Allocate output if needed
						IntPtr outputPtr = IntPtr.Zero;
						if (needsOutput)
						{
							dynamic buffer = this.MemoryH.AllocateBuffer(expectedResultType, expectedResultLength, silent);
							outputPtr = ((CUdeviceptr) buffer).Pointer;
						}

						// 5b. Prepare arguments
						CUdeviceptr inputDevicePtr = new(pointers[i]);

						// Automatische Längenbestimmung falls inputLength <= 0
						IntPtr chunkLength = inputLength;
						if (inputLength <= 0)
						{
							CudaBuffer? bufferInfo = this.MemoryH.GetBuffer(pointers[i]);
							chunkLength = bufferInfo?.Length ?? 0;
							if (chunkLength <= 0)
							{
								if (!silent)
								{
									this.Log($"Invalid buffer length for chunk {i}", "Using default length", 1);
								}
								chunkLength = expectedResultLength > 0 ? expectedResultLength : (IntPtr) 1024;
							}
						}

						CUdeviceptr outputDevicePtr = new(outputPtr);
						object[] kernelArgs = this.MergeArguments(inputDevicePtr, chunkLength, outputDevicePtr,
																expectedResultLength, channels, bitdepth,
																arguments, silent);

						// 5c. Calculate dimensions
						long totalSamples = (long) inputLength / (bitdepth / 8);
						int blockSize = 256;
						int gridSize = (int) ((totalSamples + blockSize - 1) / blockSize);

						lock (this.Kernel) // Ensure thread-safe kernel configuration
						{
							this.Kernel.BlockDimensions = new dim3(blockSize, 1, 1);
							this.Kernel.GridDimensions = new dim3(gridSize, 1, 1);
							this.Kernel.Run(kernelArgs);
						}

						// 5d. Cleanup input if we created output
						if (needsOutput)
						{
							this.MemoryH.FreeBuffer(pointers[i]);
						}

						results[i] = outputPtr != IntPtr.Zero ? outputPtr : pointers[i];

						// 5e. Update progress
						if (pBar != null)
						{
							lock (progressLock)
							{
								processed++;
								pBar.Invoke((MethodInvoker) (() =>
								{
									pBar.Value = (int) ((float) processed / pointers.Length * 100);
								}));
							}
						}
					}
					catch (Exception ex)
					{
						if (!silent)
						{
							this.Log($"Failed processing chunk {i}", ex.Message, 1);
						}

						results[i] = IntPtr.Zero;
					}
				});
			});

			// 6. Synchronize all operations
			this.Context.Synchronize();

			if (!silent)
			{
				this.Log($"Processed {pointers.Length} chunks", this.KernelName ?? "N/A", 1);
			}

			return results;
		}

		public async Task<IntPtr[]> ExecuteKernelAudioBatchAsync(IntPtr[] pointers, IntPtr inputLength, object[] arguments, IntPtr expectedResultLength = 0, Type? expectedResultType = null, int channels = 2, int bitdepth = 32, ProgressBar? pBar = null, bool silent = false)
		{
			// 1. Kernel-Check
			if (this.Kernel == null)
			{
				if (!silent)
				{
					this.Log("Kernel not loaded", this.KernelName ?? "N/A", 1);
				}

				return pointers;
			}

			// 2. Validate inputs
			if (pointers == null || pointers.Length == 0)
			{
				return [];
			}

			this.Context.SetCurrent();

			// 3. Get common parameters
			Dictionary<string, Type> args = this.GetArguments(null, silent);
			expectedResultType ??= this.MemoryH.GetBufferType(pointers[0]);
			if (expectedResultLength < 1)
			{
				expectedResultLength = this.MemoryH.GetBufferSize(pointers[0], true);
			}

			// 4. Determine if we need output buffers
			bool needsOutput = args.Count(x => x.Value == typeof(IntPtr)) == 2;
			nint[] results = new IntPtr[pointers.Length];
			int processed = 0;

			// 5. Process chunks sequentially
			for (int i = 0; i < pointers.Length; i++)
			{
				try
				{
					// 5a. Buffer length determination
					IntPtr chunkLength = inputLength;
					if (inputLength <= 0)
					{
						CudaBuffer? bufferInfo = this.MemoryH.GetBuffer(pointers[i]);
						chunkLength = bufferInfo?.Length ?? expectedResultLength;
					}

					// 5b. Allocate output if needed
					IntPtr outputPtr = IntPtr.Zero;
					if (needsOutput)
					{
						dynamic buffer = this.MemoryH.AllocateBuffer(expectedResultType, chunkLength, true);
						outputPtr = ((CUdeviceptr) buffer).Pointer;
					}

					// 5c. Prepare and run kernel (synchron in async method)
					CUdeviceptr inputDevicePtr = new(pointers[i]);
					CUdeviceptr outputDevicePtr = new(outputPtr);
					object[] kernelArgs = this.MergeArguments(inputDevicePtr, chunkLength, outputDevicePtr,
														   chunkLength, channels, bitdepth,
														   arguments, true);

					long totalSamples = (long) chunkLength / (bitdepth / 8);
					int blockSize = 256;
					int gridSize = (int) ((totalSamples + blockSize - 1) / blockSize);

					this.Kernel.BlockDimensions = new dim3(blockSize, 1, 1);
					this.Kernel.GridDimensions = new dim3(gridSize, 1, 1);
					this.Kernel.Run(kernelArgs);

					// 5d. Cleanup
					if (needsOutput)
					{
						this.MemoryH.FreeBuffer(pointers[i]);
					}

					results[i] = outputPtr != IntPtr.Zero ? outputPtr : pointers[i];
				}
				catch (Exception ex)
				{
					if (!silent)
					{
						this.Log($"Chunk {i} failed", ex.Message, 1);
					}

					results[i] = pointers[i];
				}

				// 5e. Progress update (UI-Thread)
				if (pBar != null)
				{
					await Task.Run(() =>
					{
						pBar.Invoke((MethodInvoker) (() =>
						{
							pBar.Value = ++processed * 100 / pointers.Length;
						}));
					});
				}

				await Task.Yield(); // Unterbricht den Task kurz für UI-Updates
			}

			// 6. Final sync
			this.Context.Synchronize();
			if (!silent)
			{
				this.Log($"Processed {pointers.Length} chunks", this.KernelName ?? "N/A", 1);
			}

			return results;
		}


		// Get args & merge
		public Type GetArgumentType(string typeName)
		{
			// Pointers are always IntPtr (containing *)
			if (typeName.Contains("*"))
			{
				return typeof(IntPtr);
			}

			string typeIdentifier = typeName.Split(' ').LastOrDefault()?.Trim() ?? "void";
			Type type = typeIdentifier switch
			{
				"int" => typeof(int),
				"long" => typeof(long),
				"float" => typeof(float),
				"double" => typeof(double),
				"char1" => typeof(char1),
				"bool" => typeof(bool),
				"void" => typeof(void),
				"char" => typeof(byte),
				"byte" => typeof(sbyte),
				"float2" => typeof(float2),
				_ => typeof(void)
			};

			return type;
		}

		public Dictionary<string, Type> GetArguments(string? kernelCode = null, bool silent = false)
		{
			kernelCode ??= this.KernelCode;
			if (string.IsNullOrEmpty(kernelCode) || this.Kernel == null)
			{
				if (!silent)
				{
					this.Log("Kernel code is empty", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			Dictionary<string, Type> arguments = [];

			int index = kernelCode.IndexOf("__global__ void");
			if (index == -1)
			{
				if (!silent)
				{
					this.Log($"'__global__ void' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			index = kernelCode.IndexOf("(", index);
			if (index == -1)
			{
				if (!silent)
				{
					this.Log($"'(' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			int endIndex = kernelCode.IndexOf(")", index);
			if (endIndex == -1)
			{
				if (!silent)
				{
					this.Log($"')' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			string[] args = kernelCode.Substring(index + 1, endIndex - index - 1).Split(',').Select(x => x.Trim()).ToArray();

			// Get loaded kernels function args
			for (int i = 0; i < args.Length; i++)
			{
				string name = args[i].Split(' ').LastOrDefault() ?? "N/A";
				string typeName = args[i].Replace(name, "").Trim();
				Type type = this.GetArgumentType(typeName);

				// Add to dictionary
				arguments.Add(name, type);
			}

			return arguments;
		}

		public object[] MergeArguments(CUdeviceptr inputPointer, long inputLength, CUdeviceptr outputPointer, long outputLength, int channels, int bitdepth, object[] arguments, bool silent = false)
		{
			// Get kernel argument definitions
			Dictionary<string, Type> args = this.GetArguments(null, silent);

			// Create array for kernel arguments
			object[] kernelArgs = new object[args.Count];

			int pointersCount = 0;
			// Integrate invariables if name fits (contains)
			for (int i = 0; i < kernelArgs.Length; i++)
			{
				string name = args.ElementAt(i).Key;
				Type type = args.ElementAt(i).Value;

				if (pointersCount == 0 && type == typeof(IntPtr))
				{
					kernelArgs[i] = inputPointer;
					pointersCount++;

					if (!silent)
					{
						this.Log($"In-pointer: <{inputPointer}>", "", 1);
					}
				}
				else if (pointersCount == 1 && type == typeof(IntPtr))
				{
					kernelArgs[i] = outputPointer;
					pointersCount++;

					if (!silent)
					{
						this.Log($"Out-pointer: <{outputPointer}>", "", 1);
					}
				}
				else if (name.ToLower().Contains("length") && name.ToLower().Contains("in") && (type == typeof(int) || type == typeof(long)))
				{
					kernelArgs[i] = inputLength;
					
					if (!silent)
					{
						this.Log($"Input length: [{inputLength}]", "", 1);
					}
				}
				else if (name.ToLower().Contains("length") && name.ToLower().Contains("out") && (type == typeof(int) || type == typeof(long)))
				{
					kernelArgs[i] = outputLength;

					if (!silent)
					{
						this.Log($"Output length: [{outputLength}]", "", 1);
					}
				}
				else if (name.ToLower().Contains("channel") && type == typeof(int))
				{
					kernelArgs[i] = channels;

					if (!silent)
					{
						this.Log($"Channels: [{channels}]", "", 1);
					}
				}
				else if (name.ToLower().Contains("bit") && type == typeof(int))
				{
					kernelArgs[i] = bitdepth;

					if (!silent)
					{
						this.Log($"Bitdepth: [{bitdepth}]", "", 1);
					}
				}
				else
				{
					// Check if argument is in arguments array
					for (int j = 0; j < arguments.Length; j++)
					{
						if (name == args.ElementAt(j).Key)
						{
							kernelArgs[i] = arguments[j];
							break;
						}
					}

					// If not found, set to 0
					if (kernelArgs[i] == null)
					{
						kernelArgs[i] = 0;
					}
				}
			}

			// DEBUG LOG
			this.Log("Kernel arguments: " + string.Join(", ", kernelArgs.Select(x => x.ToString())), "", 1);

			// Return kernel arguments
			return kernelArgs;
		}




	}
}