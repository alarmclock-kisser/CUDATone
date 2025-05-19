namespace CUDATone
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.listBox_tracks = new ListBox();
			this.listBox_log = new ListBox();
			this.hScrollBar_offset = new HScrollBar();
			this.pictureBox_wave = new PictureBox();
			this.numericUpDown_zoom = new NumericUpDown();
			this.button_play = new Button();
			this.textBox_time = new TextBox();
			this.label_meta = new Label();
			this.button_export = new Button();
			this.button_import = new Button();
			this.comboBox_devices = new ComboBox();
			this.progressBar_vram = new ProgressBar();
			this.progressBar_loading = new ProgressBar();
			this.comboBox_kernels = new ComboBox();
			this.button_exec = new Button();
			this.button_info = new Button();
			this.checkBox_silent = new CheckBox();
			this.numericUpDown_chunkSize = new NumericUpDown();
			this.numericUpDown_overlap = new NumericUpDown();
			this.vScrollBar_volume = new VScrollBar();
			this.checkBox_mute = new CheckBox();
			this.button_fft = new Button();
			this.button_normalize = new Button();
			this.label_info_chunkSize = new Label();
			this.label_info_overlap = new Label();
			this.groupBox_fft = new GroupBox();
			this.groupBox_controls = new GroupBox();
			this.label_info_zoom = new Label();
			this.panel_kernel = new Panel();
			this.checkBox_onlyOptionalArgs = new CheckBox();
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_zoom).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_chunkSize).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_overlap).BeginInit();
			this.groupBox_fft.SuspendLayout();
			this.groupBox_controls.SuspendLayout();
			this.SuspendLayout();
			// 
			// listBox_tracks
			// 
			this.listBox_tracks.FormattingEnabled = true;
			this.listBox_tracks.ItemHeight = 15;
			this.listBox_tracks.Location = new Point(1592, 685);
			this.listBox_tracks.Name = "listBox_tracks";
			this.listBox_tracks.Size = new Size(180, 124);
			this.listBox_tracks.TabIndex = 0;
			// 
			// listBox_log
			// 
			this.listBox_log.FormattingEnabled = true;
			this.listBox_log.ItemHeight = 15;
			this.listBox_log.Location = new Point(12, 685);
			this.listBox_log.Name = "listBox_log";
			this.listBox_log.Size = new Size(300, 124);
			this.listBox_log.TabIndex = 1;
			// 
			// hScrollBar_offset
			// 
			this.hScrollBar_offset.Location = new Point(315, 792);
			this.hScrollBar_offset.Name = "hScrollBar_offset";
			this.hScrollBar_offset.Size = new Size(1254, 17);
			this.hScrollBar_offset.TabIndex = 2;
			// 
			// pictureBox_wave
			// 
			this.pictureBox_wave.Location = new Point(318, 685);
			this.pictureBox_wave.Name = "pictureBox_wave";
			this.pictureBox_wave.Size = new Size(1251, 104);
			this.pictureBox_wave.TabIndex = 3;
			this.pictureBox_wave.TabStop = false;
			// 
			// numericUpDown_zoom
			// 
			this.numericUpDown_zoom.Location = new Point(116, 35);
			this.numericUpDown_zoom.Maximum = new decimal(new int[] { 8192, 0, 0, 0 });
			this.numericUpDown_zoom.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			this.numericUpDown_zoom.Name = "numericUpDown_zoom";
			this.numericUpDown_zoom.Size = new Size(60, 23);
			this.numericUpDown_zoom.TabIndex = 4;
			this.numericUpDown_zoom.Value = new decimal(new int[] { 256, 0, 0, 0 });
			// 
			// button_play
			// 
			this.button_play.Location = new Point(6, 92);
			this.button_play.Name = "button_play";
			this.button_play.Size = new Size(23, 23);
			this.button_play.TabIndex = 5;
			this.button_play.Text = ">";
			this.button_play.UseVisualStyleBackColor = true;
			// 
			// textBox_time
			// 
			this.textBox_time.Location = new Point(35, 92);
			this.textBox_time.Name = "textBox_time";
			this.textBox_time.PlaceholderText = "0:00:00.000";
			this.textBox_time.Size = new Size(75, 23);
			this.textBox_time.TabIndex = 6;
			// 
			// label_meta
			// 
			this.label_meta.AutoSize = true;
			this.label_meta.Location = new Point(1590, 667);
			this.label_meta.Name = "label_meta";
			this.label_meta.Size = new Size(94, 15);
			this.label_meta.TabIndex = 7;
			this.label_meta.Text = "No track loaded.";
			// 
			// button_export
			// 
			this.button_export.Location = new Point(116, 92);
			this.button_export.Name = "button_export";
			this.button_export.Size = new Size(60, 23);
			this.button_export.TabIndex = 8;
			this.button_export.Text = "Export";
			this.button_export.UseVisualStyleBackColor = true;
			this.button_export.Click += this.button_export_Click;
			// 
			// button_import
			// 
			this.button_import.Location = new Point(116, 64);
			this.button_import.Name = "button_import";
			this.button_import.Size = new Size(60, 23);
			this.button_import.TabIndex = 9;
			this.button_import.Text = "Import";
			this.button_import.UseVisualStyleBackColor = true;
			this.button_import.Click += this.button_import_Click;
			// 
			// comboBox_devices
			// 
			this.comboBox_devices.FormattingEnabled = true;
			this.comboBox_devices.Location = new Point(12, 12);
			this.comboBox_devices.Name = "comboBox_devices";
			this.comboBox_devices.Size = new Size(300, 23);
			this.comboBox_devices.TabIndex = 10;
			this.comboBox_devices.Text = "Select CUDA device to initialize ...";
			// 
			// progressBar_vram
			// 
			this.progressBar_vram.Location = new Point(12, 41);
			this.progressBar_vram.Name = "progressBar_vram";
			this.progressBar_vram.Size = new Size(300, 12);
			this.progressBar_vram.TabIndex = 11;
			// 
			// progressBar_loading
			// 
			this.progressBar_loading.Location = new Point(12, 667);
			this.progressBar_loading.Name = "progressBar_loading";
			this.progressBar_loading.Size = new Size(300, 12);
			this.progressBar_loading.TabIndex = 12;
			// 
			// comboBox_kernels
			// 
			this.comboBox_kernels.FormattingEnabled = true;
			this.comboBox_kernels.Location = new Point(12, 59);
			this.comboBox_kernels.Name = "comboBox_kernels";
			this.comboBox_kernels.Size = new Size(234, 23);
			this.comboBox_kernels.TabIndex = 13;
			this.comboBox_kernels.Text = "Select kernel to load ...";
			// 
			// button_exec
			// 
			this.button_exec.Location = new Point(252, 59);
			this.button_exec.Name = "button_exec";
			this.button_exec.Size = new Size(60, 23);
			this.button_exec.TabIndex = 14;
			this.button_exec.Text = "Exec";
			this.button_exec.UseVisualStyleBackColor = true;
			this.button_exec.Click += this.button_exec_Click;
			// 
			// button_info
			// 
			this.button_info.Location = new Point(318, 12);
			this.button_info.Name = "button_info";
			this.button_info.Size = new Size(23, 23);
			this.button_info.TabIndex = 15;
			this.button_info.Text = "i";
			this.button_info.UseVisualStyleBackColor = true;
			this.button_info.Click += this.button_info_Click;
			// 
			// checkBox_silent
			// 
			this.checkBox_silent.AutoSize = true;
			this.checkBox_silent.Location = new Point(12, 642);
			this.checkBox_silent.Name = "checkBox_silent";
			this.checkBox_silent.Size = new Size(80, 19);
			this.checkBox_silent.TabIndex = 16;
			this.checkBox_silent.Text = "Silent log?";
			this.checkBox_silent.UseVisualStyleBackColor = true;
			// 
			// numericUpDown_chunkSize
			// 
			this.numericUpDown_chunkSize.Location = new Point(103, 35);
			this.numericUpDown_chunkSize.Maximum = new decimal(new int[] { 16384, 0, 0, 0 });
			this.numericUpDown_chunkSize.Minimum = new decimal(new int[] { 128, 0, 0, 0 });
			this.numericUpDown_chunkSize.Name = "numericUpDown_chunkSize";
			this.numericUpDown_chunkSize.Size = new Size(73, 23);
			this.numericUpDown_chunkSize.TabIndex = 17;
			this.numericUpDown_chunkSize.Value = new decimal(new int[] { 2048, 0, 0, 0 });
			// 
			// numericUpDown_overlap
			// 
			this.numericUpDown_overlap.Location = new Point(103, 78);
			this.numericUpDown_overlap.Name = "numericUpDown_overlap";
			this.numericUpDown_overlap.Size = new Size(73, 23);
			this.numericUpDown_overlap.TabIndex = 18;
			this.numericUpDown_overlap.Value = new decimal(new int[] { 50, 0, 0, 0 });
			// 
			// vScrollBar_volume
			// 
			this.vScrollBar_volume.Location = new Point(1572, 685);
			this.vScrollBar_volume.Name = "vScrollBar_volume";
			this.vScrollBar_volume.Size = new Size(17, 104);
			this.vScrollBar_volume.TabIndex = 19;
			this.vScrollBar_volume.Value = 100;
			// 
			// checkBox_mute
			// 
			this.checkBox_mute.AutoSize = true;
			this.checkBox_mute.Location = new Point(35, 67);
			this.checkBox_mute.Name = "checkBox_mute";
			this.checkBox_mute.Size = new Size(59, 19);
			this.checkBox_mute.TabIndex = 20;
			this.checkBox_mute.Text = "Mute?";
			this.checkBox_mute.UseVisualStyleBackColor = true;
			// 
			// button_fft
			// 
			this.button_fft.Location = new Point(6, 78);
			this.button_fft.Name = "button_fft";
			this.button_fft.Size = new Size(73, 23);
			this.button_fft.TabIndex = 21;
			this.button_fft.Text = "(I)FFT";
			this.button_fft.UseVisualStyleBackColor = true;
			this.button_fft.Click += this.button_fft_Click;
			// 
			// button_normalize
			// 
			this.button_normalize.Location = new Point(6, 35);
			this.button_normalize.Name = "button_normalize";
			this.button_normalize.Size = new Size(75, 23);
			this.button_normalize.TabIndex = 22;
			this.button_normalize.Text = "Normalize";
			this.button_normalize.UseVisualStyleBackColor = true;
			this.button_normalize.Click += this.button_normalize_Click;
			// 
			// label_info_chunkSize
			// 
			this.label_info_chunkSize.AutoSize = true;
			this.label_info_chunkSize.Location = new Point(103, 17);
			this.label_info_chunkSize.Name = "label_info_chunkSize";
			this.label_info_chunkSize.Size = new Size(64, 15);
			this.label_info_chunkSize.TabIndex = 23;
			this.label_info_chunkSize.Text = "Chunk size";
			// 
			// label_info_overlap
			// 
			this.label_info_overlap.AutoSize = true;
			this.label_info_overlap.Location = new Point(103, 61);
			this.label_info_overlap.Name = "label_info_overlap";
			this.label_info_overlap.Size = new Size(61, 15);
			this.label_info_overlap.TabIndex = 24;
			this.label_info_overlap.Text = "Overlap %";
			// 
			// groupBox_fft
			// 
			this.groupBox_fft.Controls.Add(this.numericUpDown_overlap);
			this.groupBox_fft.Controls.Add(this.button_normalize);
			this.groupBox_fft.Controls.Add(this.label_info_overlap);
			this.groupBox_fft.Controls.Add(this.button_fft);
			this.groupBox_fft.Controls.Add(this.numericUpDown_chunkSize);
			this.groupBox_fft.Controls.Add(this.label_info_chunkSize);
			this.groupBox_fft.Location = new Point(1590, 427);
			this.groupBox_fft.Name = "groupBox_fft";
			this.groupBox_fft.Size = new Size(182, 107);
			this.groupBox_fft.TabIndex = 25;
			this.groupBox_fft.TabStop = false;
			this.groupBox_fft.Text = "CUDA FFT";
			// 
			// groupBox_controls
			// 
			this.groupBox_controls.Controls.Add(this.label_info_zoom);
			this.groupBox_controls.Controls.Add(this.button_play);
			this.groupBox_controls.Controls.Add(this.checkBox_mute);
			this.groupBox_controls.Controls.Add(this.textBox_time);
			this.groupBox_controls.Controls.Add(this.numericUpDown_zoom);
			this.groupBox_controls.Controls.Add(this.button_import);
			this.groupBox_controls.Controls.Add(this.button_export);
			this.groupBox_controls.Location = new Point(1590, 540);
			this.groupBox_controls.Name = "groupBox_controls";
			this.groupBox_controls.Size = new Size(182, 121);
			this.groupBox_controls.TabIndex = 26;
			this.groupBox_controls.TabStop = false;
			this.groupBox_controls.Text = "Controls";
			// 
			// label_info_zoom
			// 
			this.label_info_zoom.AutoSize = true;
			this.label_info_zoom.Location = new Point(116, 17);
			this.label_info_zoom.Name = "label_info_zoom";
			this.label_info_zoom.Size = new Size(39, 15);
			this.label_info_zoom.TabIndex = 7;
			this.label_info_zoom.Text = "Zoom";
			// 
			// panel_kernel
			// 
			this.panel_kernel.Location = new Point(12, 88);
			this.panel_kernel.Name = "panel_kernel";
			this.panel_kernel.Size = new Size(234, 280);
			this.panel_kernel.TabIndex = 27;
			// 
			// checkBox_onlyOptionalArgs
			// 
			this.checkBox_onlyOptionalArgs.AutoSize = true;
			this.checkBox_onlyOptionalArgs.Location = new Point(12, 374);
			this.checkBox_onlyOptionalArgs.Name = "checkBox_onlyOptionalArgs";
			this.checkBox_onlyOptionalArgs.Size = new Size(190, 19);
			this.checkBox_onlyOptionalArgs.TabIndex = 28;
			this.checkBox_onlyOptionalArgs.Text = "Show only variable arguments?";
			this.checkBox_onlyOptionalArgs.UseVisualStyleBackColor = true;
			// 
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(1784, 821);
			this.Controls.Add(this.checkBox_onlyOptionalArgs);
			this.Controls.Add(this.panel_kernel);
			this.Controls.Add(this.groupBox_controls);
			this.Controls.Add(this.groupBox_fft);
			this.Controls.Add(this.vScrollBar_volume);
			this.Controls.Add(this.checkBox_silent);
			this.Controls.Add(this.button_info);
			this.Controls.Add(this.button_exec);
			this.Controls.Add(this.comboBox_kernels);
			this.Controls.Add(this.progressBar_loading);
			this.Controls.Add(this.progressBar_vram);
			this.Controls.Add(this.comboBox_devices);
			this.Controls.Add(this.label_meta);
			this.Controls.Add(this.pictureBox_wave);
			this.Controls.Add(this.hScrollBar_offset);
			this.Controls.Add(this.listBox_log);
			this.Controls.Add(this.listBox_tracks);
			this.Name = "WindowMain";
			this.Text = "CUDATone";
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_zoom).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_chunkSize).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_overlap).EndInit();
			this.groupBox_fft.ResumeLayout(false);
			this.groupBox_fft.PerformLayout();
			this.groupBox_controls.ResumeLayout(false);
			this.groupBox_controls.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();
		}

		#endregion

		private ListBox listBox_tracks;
		private ListBox listBox_log;
		private HScrollBar hScrollBar_offset;
		private PictureBox pictureBox_wave;
		private NumericUpDown numericUpDown_zoom;
		private Button button_play;
		private TextBox textBox_time;
		private Label label_meta;
		private Button button_export;
		private Button button_import;
		private ComboBox comboBox_devices;
		private ProgressBar progressBar_vram;
		private ProgressBar progressBar_loading;
		private ComboBox comboBox_kernels;
		private Button button_exec;
		private Button button_info;
		private CheckBox checkBox_silent;
		private NumericUpDown numericUpDown_chunkSize;
		private NumericUpDown numericUpDown_overlap;
		private VScrollBar vScrollBar_volume;
		private CheckBox checkBox_mute;
		private Button button_fft;
		private Button button_normalize;
		private Label label_info_chunkSize;
		private Label label_info_overlap;
		private GroupBox groupBox_fft;
		private GroupBox groupBox_controls;
		private Label label_info_zoom;
		private Panel panel_kernel;
		private CheckBox checkBox_onlyOptionalArgs;
	}
}
