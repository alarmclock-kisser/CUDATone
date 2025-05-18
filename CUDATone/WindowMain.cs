namespace CUDATone
{
	public partial class WindowMain : Form
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		public string Repopath { get; set; } = string.Empty;

		public AudioHandling AH;






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
		}






		// ----- ----- ----- METHODS ----- ----- ----- \\






		// ----- ----- ----- EVENTS ----- ----- ----- \\
		private void button_import_Click(object sender, EventArgs e)
		{
			this.AH.Import();
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			string? result = this.AH.CurrentObject?.Export();
			if (!string.IsNullOrEmpty(result))
			{
				MessageBox.Show(result, "Exported!", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
		}





	}
}
