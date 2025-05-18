namespace CUDATone
{
	public partial class WindowMain : Form
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		public string Repopath { get; set; } = string.Empty;

		public AudioHandling AH;
		public CudaContextHandling CudaH;





		// ----- ----- ----- CONSTRUCTORS ----- ----- ----- \\
		public WindowMain()
		{
			InitializeComponent();

			// Set repopath
			this.Repopath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\"));

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Init. classes
			this.AH = new AudioHandling(this.Repopath, this.listBox_log, this.listBox_tracks, this.pictureBox_wave, this.hScrollBar_offset, this.button_play, this.textBox_time, this.label_meta, this.numericUpDown_zoom);
			this.CudaH = new CudaContextHandling(this.Repopath, this.listBox_log, this.comboBox_devices, this.comboBox_kernels, this.progressBar_vram);

			// Register events
			this.listBox_tracks.DoubleClick += (s, e) => this.MoveTrack(this.listBox_tracks.SelectedIndex);

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
		public IntPtr? MoveTrack(int index = -1)
		{
			// If index is -1: Get CurrentObject index
			if (index == -1)
			{
				if (this.AH.CurrentObject == null)
				{
					return null;
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
				return null;
			}

			// Check initialized
			if (this.CudaH.MemoryH == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("CUDA not initialized", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return null;
			}

			// Get track
			AudioObject track = this.AH.Tracks[index];
			if (track == null)
			{
				if (!this.checkBox_silent.Checked)
				{
					MessageBox.Show("Couldn't get track", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return null;
			}

			// Move tracks floats between Host <-> CUDA
			if (track.OnHost)
			{
				

			}
			else if (track.OnDevice)
			{
				
			}

			// Failed
			this.AH.UpdateView();
			return null;
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

		
	}
}
