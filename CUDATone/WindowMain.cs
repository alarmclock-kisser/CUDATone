using ManagedCuda.VectorTypes;

namespace CUDATone
{
	public partial class WindowMain : Form
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		public string Repopath { get; set; } = string.Empty;

		public AudioHandling AH;
		public CudaContextHandling CudaH;
		public GuiBuilder GuiB;




		private bool isProcessing;
		private Dictionary<NumericUpDown, int> previousNumericValues = [];


		// ----- ----- ----- CONSTRUCTORS ----- ----- ----- \\
		public WindowMain()
		{
			this.InitializeComponent();

			// Set repopath
			this.Repopath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\"));

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Init. classes
			this.AH = new AudioHandling(this.Repopath, this.listBox_log, this.listBox_tracks, this.pictureBox_wave, this.hScrollBar_offset, this.button_play, this.textBox_time, this.label_meta, this.numericUpDown_zoom, this.vScrollBar_volume, this.checkBox_mute);
			this.CudaH = new CudaContextHandling(this.Repopath, this.listBox_log, this.comboBox_devices, this.comboBox_kernels, this.progressBar_vram);
			this.GuiB = new GuiBuilder(this.Repopath, this.listBox_log, this.CudaH, this.AH, this.panel_kernel, this.checkBox_silent);


			// Register events
			this.listBox_tracks.DoubleClick += (s, e) => this.MoveTrack(this.listBox_tracks.SelectedIndex);
			this.RegisterNumericToSecondPow(this.numericUpDown_zoom);
			this.RegisterNumericToSecondPow(this.numericUpDown_chunkSize);
			this.comboBox_kernels.SelectedIndexChanged += (s, e) => this.LoadKernel(this.comboBox_kernels.SelectedItem?.ToString() ?? "");
			this.checkBox_onlyOptionalArgs.CheckedChanged += (s, e) => this.GuiB.BuildPanel(0.6f, this.button_exec, this.checkBox_onlyOptionalArgs.Checked);


			// Select first CUDA device
			if (this.comboBox_devices.Items.Count > 0)
			{
				this.comboBox_devices.SelectedIndex = 0;
			}
			else
			{
				this.comboBox_devices.Enabled = false;
				this.button_info.Enabled = false;
			}

		}






		// ----- ----- ----- METHODS ----- ----- ----- \\
		public void RegisterNumericToSecondPow(NumericUpDown numeric)
		{
			// Initialwert speichern
			this.previousNumericValues.Add(numeric, (int) numeric.Value);

			numeric.ValueChanged += (s, e) =>
			{
				// Verhindere rekursive Aufrufe
				if (this.isProcessing)
				{
					return;
				}

				this.isProcessing = true;

				try
				{
					int newValue = (int) numeric.Value;
					int oldValue = this.previousNumericValues[numeric];
					int max = (int) numeric.Maximum;
					int min = (int) numeric.Minimum;

					// Nur verarbeiten, wenn sich der Wert tats chlich ge ndert hat
					if (newValue != oldValue)
					{
						int calculatedValue;

						if (newValue > oldValue)
						{
							// Verdoppeln, aber nicht  ber Maximum
							calculatedValue = Math.Min(oldValue * 2, max);
						}
						else if (newValue < oldValue)
						{
							// Halbieren, aber nicht unter Minimum
							calculatedValue = Math.Max(oldValue / 2, min);
						}
						else
						{
							calculatedValue = oldValue;
						}

						// Nur aktualisieren wenn notwendig
						if (calculatedValue != newValue)
						{
							numeric.Value = calculatedValue;
						}

						this.previousNumericValues[numeric] = calculatedValue;
					}
				}
				finally
				{
					this.isProcessing = false;
				}
			};
		}

		public void MoveTrack(int index = -1)
		{
			// If index is -1: Get CurrentObject index
			if (index == -1)
			{
				if (this.AH.CurrentObject == null)
				{
					return;
				}

				index = this.AH.Tracks.IndexOf(this.AH.CurrentObject);
			}

			// Check index range
			if (index < 0 || index >= this.AH.Tracks.Count)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Invalid track id", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Check initialized
			if (this.CudaH.MemoryH == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("CUDA not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Get track
			AudioObject track = this.AH.Tracks[index];
			if (track == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get track", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Move tracks floats between Host <-> CUDA
			if (track.OnHost)
			{
				// Get chunks from track
				int chunkSize = (int) this.numericUpDown_chunkSize.Value;
				float overlap = (float) this.numericUpDown_overlap.Value / 100f;
				List<Single[]> chunks = track.GetChunks(chunkSize, overlap);
				if (chunks == null || chunks.Count == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't get chunks", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}

				// Move chunks to CUDA
				track.Pointers = this.CudaH.MemoryH.PushChunks<float>(chunks, this.checkBox_silent.Checked);
				if (track.Pointers == null || track.Pointers.Length == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't push chunks", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}

				// Void track data
				track.Data = [];
				this.AH.UpdateView();

				return;
			}
			else if (track.OnDevice)
			{
				// Get Pointers from track
				IntPtr[] pointers = track.Pointers;
				if (pointers == null || pointers.Length == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't get pointers", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}

				// Move chunks to Host (track) + free
				track.AggregateChunks(this.CudaH.MemoryH.PullChunks<float>(pointers, true, this.checkBox_silent.Checked));
				if (track.Data == null || track.Data.Length == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't pull chunks", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}

				// Void track pointers
				track.Pointers = [];
				this.AH.UpdateView();

				return;
			}

			// Failed
			this.AH.UpdateView();
			return;
		}

		public async void PerformFft(int index = -1)
		{
			// If index is -1: Get CurrentObject index
			if (index == -1)
			{
				if (this.AH.CurrentObject == null)
				{
					return;
				}

				index = this.AH.Tracks.IndexOf(this.AH.CurrentObject);
			}

			// Check index range
			if (index < 0 || index >= this.AH.Tracks.Count)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Invalid track id", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Check initialized
			if (this.CudaH.MemoryH == null || this.CudaH.FourierH == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("CUDA not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Get track
			AudioObject track = this.AH.Tracks[index];
			if (track == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get track", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Verify track on device
			bool moved = false;
			if (!track.OnDevice)
			{
				this.MoveTrack(this.AH.Tracks.IndexOf(track));
				moved = true;

				// Abort if still not on device
				if (!track.OnDevice)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't move track to CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}
			}

			IntPtr[] pointers = track.Pointers;
			Type type = this.CudaH.MemoryH.GetBufferType(pointers.FirstOrDefault());

			if (type == typeof(void))
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get buffer type", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}
			else if (type == typeof(float))
			{
				// Perform FFT on device
				track.Pointers = await this.CudaH.FourierH.ExecuteFFTAsync(pointers, true, this.checkBox_silent.Checked, this.progressBar_loading);
			}
			else if (type == typeof(float2))
			{
				// Perform FFT on device
				track.Pointers = await this.CudaH.FourierH.ExecuteIFFTAsync(pointers, true, this.checkBox_silent.Checked, this.progressBar_loading);
			}
			else
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Unsupported buffer type", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Reset progress bar
			this.progressBar_loading.Value = 0;

			// Optionally move back to host
			if (moved && track.OnDevice)
			{
				this.MoveTrack(this.AH.Tracks.IndexOf(track));
			}

			// Refresh UI
			this.AH.UpdateView();
		}

		public void LoadKernel(string kernelName = "")
		{
			// Load kernel
			this.CudaH.KernelH?.LoadKernel(kernelName, this.checkBox_silent.Checked);
			if (this.CudaH.KernelH?.Kernel == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Get arguments
			this.GuiB.BuildPanel(0.6f, this.button_exec, this.checkBox_onlyOptionalArgs.Checked);

		}

		public async void ExecKernelChunksOld(int index = -1)
		{
			// If index is -1: Get CurrentObject index
			if (index == -1)
			{
				if (this.AH.CurrentObject == null)
				{
					return;
				}

				index = this.AH.Tracks.IndexOf(this.AH.CurrentObject);
			}

			// Check index range
			if (index < 0 || index >= this.AH.Tracks.Count)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Invalid track id", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Check initialized
			if (this.CudaH.MemoryH == null || this.CudaH.KernelH == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("CUDA not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Get track
			AudioObject track = this.AH.Tracks[index];
			if (track == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get track", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Verify track on device
			bool moved = false;
			if (!track.OnDevice)
			{
				this.MoveTrack(this.AH.Tracks.IndexOf(track));
				moved = true;

				// Abort if still not on device
				if (!track.OnDevice)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Couldn't move track to CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}
					return;
				}
			}

			IntPtr[] pointers = track.Pointers;
			Type type = this.CudaH.MemoryH.GetBufferType(pointers.FirstOrDefault());
			if (type == typeof(void))
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get buffer type", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Get kernel arguments
			object[] arguments = this.GuiB.GetArgumentValues();

			// Execute kernel batch async
			IntPtr[] result = await this.CudaH.KernelH.ExecuteKernelAudioBatchAsync(pointers, IntPtr.Zero, arguments, IntPtr.Zero, type, track.Channels, track.Bitdepth, this.progressBar_loading, this.checkBox_silent.Checked);
			if (result == null || result.Length == 0)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Failed to execute kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}

			// Set result to track
			track.Pointers = result;

			// Reset progress bar
			this.progressBar_loading.Value = 0;

			// Optionally move back to host
			if (moved && track.OnDevice)
			{
				this.MoveTrack(this.AH.Tracks.IndexOf(track));
			}

			// Refresh UI
			this.AH.UpdateView();
		}

		public async Task ExecKernelChunks(int index = -1)
		{
			try
			{
				// 1. Initiale Validierungen
				if (index == -1)
				{
					if (this.AH.CurrentObject == null)
					{
						if (!this.checkBox_silent.Checked)
						{
							MessageBox.Show("No track selected", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
						}

						return;
					}
					index = this.AH.Tracks.IndexOf(this.AH.CurrentObject);
				}

				if (index < 0 || index >= this.AH.Tracks.Count)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Invalid track index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				// 2. CUDA-Initialisierung pr�fen
				if (this.CudaH?.MemoryH == null || this.CudaH?.KernelH == null)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("CUDA not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				// 3. Track validieren
				AudioObject track = this.AH.Tracks[index];
				if (track == null)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Track not found", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				// 4. Auf Device verschieben falls n�tig
				bool movedToDevice = false;
				if (!track.OnDevice)
				{
					await Task.Run(() => this.MoveTrack(index));
					if (!track.OnDevice)
					{
						if (!this.checkBox_silent.Checked)
						{
							MessageBox.Show("Failed to move track to device", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
						}

						return;
					}
					movedToDevice = true;
				}

				// 5. Pointer und Typ validieren
				if (track.Pointers == null || track.Pointers.Length == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("No device pointers available", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				Type bufferType = this.CudaH.MemoryH.GetBufferType(track.Pointers[0]);
				if (bufferType == typeof(void))
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Invalid buffer type", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				// 6. Kernel ausf�hren
				Object[] arguments = this.GuiB.GetArgumentValues();
				this.progressBar_loading.Value = 0;

				IntPtr[] results = await this.CudaH.KernelH.ExecuteKernelAudioBatchAsync(
					track.Pointers,
					IntPtr.Zero, // Automatische L�ngenbestimmung
					arguments,
					IntPtr.Zero, // Erwartete L�nge
					bufferType,
					track.Channels,
					track.Bitdepth,
					this.progressBar_loading,
					this.checkBox_silent.Checked);

				if (results == null || results.Length == 0)
				{
					if (!this.checkBox_silent.Checked)
					{
						MessageBox.Show("Kernel execution failed", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					}

					return;
				}

				// 7. Ergebnisse verarbeiten
				track.Pointers = results;

				// Optional: Zur�ck zum Host
				if (movedToDevice && track.OnDevice)
				{
					await Task.Run(() => this.MoveTrack(index));
				}

				// UI aktualisieren
				this.AH.UpdateView();
			}
			catch (Exception ex)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show($"Error: {ex.Message}", "Critical Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
			finally
			{
				this.progressBar_loading.Value = 0;
			}
		}


		// ----- ----- ----- EVENTS ----- ----- ----- \\
		private void button_info_Click(object sender, EventArgs e)
		{
			List<string> info = this.CudaH.GetInfo();

			if (info.Count == 0)
			{
				MessageBox.Show("No info available", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Format each line to add newline after colon and double newline between items
			string message = string.Join(Environment.NewLine + Environment.NewLine,
				info.Select(item =>
				{
					// Split at first colon to add newline
					int colonIndex = item.IndexOf(':');
					if (colonIndex > 0)
					{
						return item.Insert(colonIndex + 1, Environment.NewLine);
					}
					return item;
				}));

			MessageBox.Show(message, "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
		}

		private void button_import_Click(object sender, EventArgs e)
		{
			this.AH.Import(this.checkBox_silent.Checked);
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			string? result = this.AH.CurrentObject?.Export();
			if (!string.IsNullOrEmpty(result))
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show(result, "Exported!", MessageBoxButtons.OK, MessageBoxIcon.Information);
				}
			}
		}

		private void button_fft_Click(object sender, EventArgs e)
		{
			this.PerformFft();
		}

		private void button_normalize_Click(object sender, EventArgs e)
		{
			this.AH.Normalize();

			this.AH.UpdateView();
		}

		private async void button_exec_Click(object sender, EventArgs e)
		{
			await this.ExecKernelChunks();
		}
	}
}
