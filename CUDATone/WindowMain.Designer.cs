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
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(1784, 821);
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
	}
}
