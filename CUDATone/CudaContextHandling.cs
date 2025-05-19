
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace CUDATone
{
	public class CudaContextHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private ComboBox DevicesCombo;
		public ComboBox KernelsCombo;
		public ProgressBar VramBar;

		public int Index = -1;
		public CUdevice? Device = null;
		public PrimaryContext? Context = null;

		public CudaMemoryHandling? MemoryH;
		public CudaFourierHandling? FourierH;
		public CudaKernelHandling? KernelH;

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaContextHandling(string repopath, ListBox listBox_log, ComboBox comboBox_devices, ComboBox comboBox_kernels, ProgressBar? progressBar_Vram = null)
		{
			this.Repopath = repopath;
			this.LogList = listBox_log;
			this.DevicesCombo = comboBox_devices;
			this.KernelsCombo = comboBox_kernels;
			this.VramBar = progressBar_Vram ?? new ProgressBar();

			// Register events
			this.DevicesCombo.SelectedIndexChanged += (s, e) => this.InitDevice(this.DevicesCombo.SelectedIndex);

			// Fill devices combobox
			this.FillDevicesCombobox();

		}




		// ----- ----- METHODS ----- ----- \\
		public void Log(string message = "", string inner = "", int indent = 0)
		{
			string msg = $"[Context]: {new string('~', indent)}{message}{(string.IsNullOrEmpty(inner) ? "" : $" ({inner})")}";

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



		public int GetDeviceCount()
		{
			// Trycatch

			int deviceCount = 0;

			try
			{
				deviceCount = CudaContext.GetDeviceCount();
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device count", ex.Message, 1);
			}

			return deviceCount;
		}

		public List<CUdevice> GetDevices()
		{
			List<CUdevice> devices = [];
			int deviceCount = this.GetDeviceCount();

			for (int i = 0; i < deviceCount; i++)
			{
				// Trycatch
				try
				{
					CUdevice device = new(i);
					devices.Add(device);
				}
				catch (CudaException ex)
				{
					this.Log("Couldn't get device # " + i, ex.Message, 1);
				}
				catch (Exception ex)
				{
					this.Log("Couldn't get device # " + i, ex.Message, 1);
				}
				finally
				{
					if (devices.Count == 0)
					{
						this.Log("No devices found", "", 1);
					}
				}
			}

			return devices;
		}

		public Version GetCapability(int index = -1)
		{
			index = index == -1 ? this.Index : index;

			Version ver = new(0, 0);

			try
			{
				ver = CudaContext.GetDeviceComputeCapability(index);
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device capability", ex.Message, 1);
			}

			return ver;
		}

		public string GetName(int index = -1)
		{
			index = index == -1 ? this.Index : index;

			string name = "N/A";

			try
			{
				name = CudaContext.GetDeviceName(index);
			}
			catch (CudaException ex)
			{
				this.Log("Couldn't get device name", ex.Message, 1);
			}

			return name;
		}

		public void FillDevicesCombobox(ComboBox? comboBox = null)
		{
			comboBox ??= this.DevicesCombo;
			comboBox.Items.Clear();

			List<CUdevice> devices = this.GetDevices();
			for (int i = 0; i < devices.Count; i++)
			{
				CUdevice device = devices[i];
				string deviceName = CudaContext.GetDeviceName(i);
				Version capability = this.GetCapability(i);
				comboBox.Items.Add($"{deviceName} ({capability.Major}.{capability.Minor})");
			}
		}

		public void InitDevice(int index = -1)
		{
			this.Dispose();

			index = index == -1 ? this.Index : index;
			if (index < 0 || index >= this.GetDeviceCount())
			{
				this.Log("Invalid device id", "Out of range");
				return;
			}			

			this.Index = index;
			this.Device = new CUdevice(index);
			this.Context = new PrimaryContext(this.Device.Value);
			this.Context.SetCurrent();
			this.MemoryH = new CudaMemoryHandling(this.Repopath, this.LogList, this.Context, this.VramBar);
			this.FourierH = new CudaFourierHandling(this.Repopath, this.LogList, this.Context, this.MemoryH);
			this.KernelH = new CudaKernelHandling(this.Repopath, this.LogList, this.Context, this.MemoryH, this.KernelsCombo);

			this.Log($"Initialized #{index}", this.GetName().Split(' ').FirstOrDefault() ?? "N/A");

		}

		public void Dispose()
		{
			this.Context?.Dispose();
			this.Context = null;
			this.Device = null;
			this.MemoryH?.Dispose();
			this.MemoryH = null;
			this.FourierH?.Dispose();
			this.FourierH = null;
			this.KernelH?.Dispose();
			this.KernelH = null;
		}

		public List<string> GetInfo()
		{
			// Abort if not initialized
			if (this.Context == null || this.Device == null || this.MemoryH == null || this.KernelH == null)
			{
				this.Log("Context not initialized", "Abort", 1);
				return [];
			}

			List<string> info = [];

			// Add name
			info.Add($"Name: {this.GetName()}");

			// Add capability
			info.Add($"Capability: {this.GetCapability().Major}.{this.GetCapability().Minor}");

			// Add memory
			info.Add($"Total memory: {this.MemoryH.GetTotalMemory(true)} MB");
			info.Add($"Used memory: {this.MemoryH.GetTotalMemoryUsage(true, true)} MB");
			info.Add($"Free memory: {this.MemoryH.GetFreeMemory(true)} MB");

			// Add pcie
			info.Add($"PCIe index: {this.Context.GetDeviceInfo().PCIDomainID}");

			// Add ecc
			info.Add($"ECC enabled: {this.Context.GetDeviceInfo().EccEnabled}");

			// Add bus
			info.Add($"Bus width: {this.Context.GetDeviceInfo().GlobalMemoryBusWidth} bits");

			// Add block
			info.Add($"Max. block dim: {this.Context.GetDeviceInfo().MaxBlockDim}");

			// Add grid
			info.Add($"Max. grid dim: {this.Context.GetDeviceInfo().MaxGridDim}");

			// Add blocks multi
			info.Add($"Max. blocks / multi: {this.Context.GetDeviceInfo().MaxBlocksPerMultiProcessor}");

			// Add async engines
			info.Add($"Async engines: {this.Context.GetDeviceInfo().AsyncEngineCount}");

			// Add threads block
			info.Add($"Max. threads / block: {this.Context.GetDeviceInfo().MaxThreadsPerBlock}");

			// Add threads multi
			info.Add($"Max. threads / multi: {this.Context.GetDeviceInfo().MaxThreadsPerMultiProcessor}");

			// Add clock
			info.Add($"Clock rate: {this.Context.GetDeviceInfo().ClockRate / 1000} MHz");

			// Add mem clock
			info.Add($"Memory clock rate: {this.Context.GetDeviceInfo().MemoryClockRate / 2000} MHz");

			// Add cores
			info.Add($"Cores: {this.Context.GetDeviceInfo().MultiProcessorCount}");

			// Add driver
			info.Add($"Driver version: {this.Context.GetDeviceInfo().DriverVersion}");

			return info;
		}

	}
}