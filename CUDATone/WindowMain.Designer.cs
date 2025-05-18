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
			this.button1 = new Button();
			this.button_info = new Button();
			this.checkBox_silent = new CheckBox();
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_zoom).BeginInit();
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
			this.hScrollBar_offset.Size = new Size(1274, 17);
			this.hScrollBar_offset.TabIndex = 2;
			// 
			// pictureBox_wave
			// 
			this.pictureBox_wave.Location = new Point(318, 685);
			this.pictureBox_wave.Name = "pictureBox_wave";
			this.pictureBox_wave.Size = new Size(1268, 104);
			this.pictureBox_wave.TabIndex = 3;
			this.pictureBox_wave.TabStop = false;
			// 
			// numericUpDown_zoom
			// 
			this.numericUpDown_zoom.Location = new Point(1712, 641);
			this.numericUpDown_zoom.Maximum = new decimal(new int[] { 8192, 0, 0, 0 });
			this.numericUpDown_zoom.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			this.numericUpDown_zoom.Name = "numericUpDown_zoom";
			this.numericUpDown_zoom.Size = new Size(60, 23);
			this.numericUpDown_zoom.TabIndex = 4;
			this.numericUpDown_zoom.Value = new decimal(new int[] { 256, 0, 0, 0 });
			// 
			// button_play
			// 
			this.button_play.Location = new Point(1590, 641);
			this.button_play.Name = "button_play";
			this.button_play.Size = new Size(23, 23);
			this.button_play.TabIndex = 5;
			this.button_play.Text = ">";
			this.button_play.UseVisualStyleBackColor = true;
			// 
			// textBox_time
			// 
			this.textBox_time.Location = new Point(1619, 641);
			this.textBox_time.Name = "textBox_time";
			this.textBox_time.PlaceholderText = "0:00:00.000";
			this.textBox_time.Size = new Size(87, 23);
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
			this.button_export.Location = new Point(1712, 612);
			this.button_export.Name = "button_export";
			this.button_export.Size = new Size(60, 23);
			this.button_export.TabIndex = 8;
			this.button_export.Text = "Export";
			this.button_export.UseVisualStyleBackColor = true;
			this.button_export.Click += this.button_export_Click;
			// 
			// button_import
			// 
			this.button_import.Location = new Point(1712, 583);
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
			// button1
			// 
			this.button1.Location = new Point(252, 59);
			this.button1.Name = "button1";
			this.button1.Size = new Size(60, 23);
			this.button1.TabIndex = 14;
			this.button1.Text = "button1";
			this.button1.UseVisualStyleBackColor = true;
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
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(1784, 821);
			this.Controls.Add(this.checkBox_silent);
			this.Controls.Add(this.button_info);
			this.Controls.Add(this.button1);
			this.Controls.Add(this.comboBox_kernels);
			this.Controls.Add(this.progressBar_loading);
			this.Controls.Add(this.progressBar_vram);
			this.Controls.Add(this.comboBox_devices);
			this.Controls.Add(this.button_import);
			this.Controls.Add(this.button_export);
			this.Controls.Add(this.label_meta);
			this.Controls.Add(this.textBox_time);
			this.Controls.Add(this.button_play);
			this.Controls.Add(this.numericUpDown_zoom);
			this.Controls.Add(this.pictureBox_wave);
			this.Controls.Add(this.hScrollBar_offset);
			this.Controls.Add(this.listBox_log);
			this.Controls.Add(this.listBox_tracks);
			this.Name = "WindowMain";
			this.Text = "CUDATone";
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_zoom).EndInit();
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
		private Button button1;
		private Button button_info;
		private CheckBox checkBox_silent;
	}
}
