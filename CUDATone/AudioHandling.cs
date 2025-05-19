using NAudio.Wave;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Linq;
using System.Threading;
using NAudio.Utils;

namespace CUDATone
{
	public class AudioHandling
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private ListBox TrackList;
		private PictureBox WavePBox;
		private HScrollBar OffsetScroll;
		private Button PlayButton;
		private TextBox TimeText;
		private Label MetaLabel;
		private NumericUpDown ZoomNumeric;
		private VScrollBar VolumeScroll;
		private CheckBox MuteCheck;

		public Color GraphColor {  get; set; } = Color.FromName("HotTrack");
		public Color BackColor
		{
			set
			{
				this.WavePBox.BackColor = value;
			}
			get
			{
				return this.WavePBox.BackColor;
			}
		}

		public List<AudioObject> Tracks = [];
		private bool isPlaying = false;
		private CancellationTokenSource playbackCancellation;
		private int oldZoomValue = 0;
		private bool isMuted = false;
		private float lastVolume = 0.8f;

		// ----- ----- ----- CONSTRUCTOR ----- ----- ----- \\
		public AudioHandling(string repopath, ListBox listBox_log, ListBox listBox_tracks,
							PictureBox pictureBox_waveform, HScrollBar hScrollBar_offset,
							Button button_playback, TextBox textBox_timestamp,
							Label label_meta, NumericUpDown numericUpDown_zoom,
							VScrollBar vScrollBar_volume, CheckBox checkBox_mute)
		{
			this.Repopath = repopath;
			this.LogList = listBox_log;
			this.TrackList = listBox_tracks;
			this.WavePBox = pictureBox_waveform;
			this.OffsetScroll = hScrollBar_offset;
			this.PlayButton = button_playback;
			this.TimeText = textBox_timestamp;
			this.MetaLabel = label_meta;
			this.ZoomNumeric = numericUpDown_zoom;
			this.VolumeScroll = vScrollBar_volume;
			this.MuteCheck = checkBox_mute;
			this.playbackCancellation = new CancellationTokenSource();

			// Initialize events
			this.PlayButton.Click += (s, e) => this.TogglePlayback();
			this.OffsetScroll.Scroll += (s, e) => this.UpdateWaveform();
			this.ZoomNumeric.ValueChanged += (s, e) => this.UpdateWaveform();
			this.TrackList.SelectedIndexChanged += (s, e) => this.UpdateTrackInfo();
			this.TrackList.SelectedIndexChanged += (s, e) => this.UpdateView();
			this.ZoomNumeric.ValueChanged += (s, e) => this.ToggleZoom();
			this.WavePBox.MouseWheel += (s, e) =>
			{
				if (e.Delta < 0)
				{
					this.ZoomNumeric.Value = Math.Min(this.ZoomNumeric.Maximum, this.ZoomNumeric.Value + 1);
				}
				else
				{
					this.ZoomNumeric.Value = Math.Max(this.ZoomNumeric.Minimum, this.ZoomNumeric.Value - 1);
				}
			};
			this.TrackList.MouseDown += (s, e) =>
			{
				if (Control.MouseButtons == MouseButtons.Right)
				{
					this.RemoveTrack(this.TrackList.SelectedIndex);
				}
			};

			// Volume-Scroll initialisieren
			this.VolumeScroll = vScrollBar_volume;
			this.VolumeScroll.Minimum = 0;
			this.VolumeScroll.Maximum = 100;
			this.VolumeScroll.Value = 80; // Standard-Lautstärke (20%)
			this.VolumeScroll.Scroll += (s, e) => this.UpdateVolume();

			// Mute-Checkbox
			this.MuteCheck.CheckedChanged += (s, e) => this.ToggleMute();

			// Initiale Lautstärke setzen (direkt beim Start)
			this.lastVolume = (100 - this.VolumeScroll.Value) / 100f; // 0.0f (leise) bis 1.0f (laut)

			// Initialize UI
			this.oldZoomValue = (int) this.ZoomNumeric.Value;
			this.UpdateTrackList();
			this.ImportResourcesAudio(true);
		}

		// ----- ----- ----- PROPERTIES ----- ----- ----- \\
		public AudioObject? CurrentObject
		{
			get
			{
				if (this.TrackList.InvokeRequired)
				{
					return (AudioObject?) this.TrackList.Invoke(new Func<AudioObject?>(() =>
					{
						return (this.TrackList.SelectedIndex >= 0 && this.TrackList.SelectedIndex < this.Tracks.Count) ?
							   this.Tracks[this.TrackList.SelectedIndex] : null;
					}));
				}
				else
				{
					return (this.TrackList.SelectedIndex >= 0 && this.TrackList.SelectedIndex < this.Tracks.Count) ?
						   this.Tracks[this.TrackList.SelectedIndex] : null;
				}
			}
		}

		public Image CurrentWaveform =>
			this.CurrentObject?.DrawWaveformParallel(this.WavePBox, (int) this.ZoomNumeric.Value,
												   (long) this.OffsetScroll.Value, this.GraphColor) ??
			new Bitmap(this.WavePBox.Width, this.WavePBox.Height);

		// ----- ----- ----- PUBLIC METHODS ----- ----- ----- \\
		public void Log(string message = "", string inner = "", int indent = 0)
		{
			string msg = $"[Audio]: {new string('~', indent)}{message}{(string.IsNullOrEmpty(inner) ? "" : $" ({inner})")}";

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

		public void AddTrack(string filepath, bool silent = false)
		{
			if (string.IsNullOrEmpty(filepath))
			{
				return;
			}

			try
			{
				AudioObject audioObject = new(filepath);
				this.Tracks.Add(audioObject);
				this.UpdateTrackList();
				this.UpdateWaveform();

				if (!silent)
				{
					this.Log($"Track added: {Path.GetFileName(filepath)}");
				}
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					this.Log("Error adding track", ex.Message, 1);
				}
			}

			// Select last entry
			if (this.TrackList.InvokeRequired)
			{
				this.TrackList.Invoke((MethodInvoker) (() =>
				{
					this.TrackList.SelectedIndex = this.Tracks.Count - 1;
				}));
			}
			else
			{
				this.TrackList.SelectedIndex = this.Tracks.Count - 1;
			}
			// Refresh view
			this.UpdateWaveform();
		}

		public string? Import(bool silent = false)
		{
			using OpenFileDialog ofd = new();
			ofd.Title = "Import audio file";
			ofd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
			ofd.Filter = "Audio files (*.wav;*.mp3)|*.wav;*.mp3";
			ofd.Multiselect = false;

			if (ofd.ShowDialog() == DialogResult.OK)
			{
				this.AddTrack(ofd.FileName, silent);
				return ofd.FileName;
			}
			return null;
		}

		public void RemoveTrack(int index = -1)
		{
			if (index == -1)
			{
				index = this.TrackList.SelectedIndex;
			}
			if (index < 0 || index >= this.Tracks.Count)
			{
				return;
			}

			this.Tracks.RemoveAt(index);
			this.UpdateTrackList();
			this.UpdateWaveform();
		}

		public void UpdateView()
		{
			if (this.CurrentObject == null)
			{
				return;
			}

			this.UpdateTrackInfo();
			this.UpdateWaveform();
		}

		public void ToggleMute()
		{
			if (this.CurrentObject == null)
			{
				return;
			}

			this.isMuted = !this.isMuted;
			if (this.isMuted)
			{
				this.MuteCheck.Text = "Muted";
				this.CurrentObject.Player.Volume = 0;
			}
			else
			{
				this.MuteCheck.Text = "Mute?";
				this.CurrentObject.Player.Volume = this.lastVolume;
			}
		}

		public void Normalize(int index = -1, float maxAmplitude = 1.0f)
		{
			if (index == -1)
			{
				index = this.TrackList.SelectedIndex;
			}
			if (index < 0 || index >= this.Tracks.Count)
			{
				return;
			}

			this.Tracks[index].Normalize(maxAmplitude);
		}

		// ----- ----- ----- PRIVATE METHODS ----- ----- ----- \\
		private void UpdateVolume()
		{
			if (this.CurrentObject == null || this.CurrentObject.Player == null)
			{
				return;
			}

			// Invertiert: 100 (Scroll ganz unten) → 0.0f (leise), 0 (ganz oben) → 1.0f (laut)
			float volume = (100 - this.VolumeScroll.Value) / 100f;

			// Nur aktualisieren, wenn nicht stummgeschaltet
			if (!this.isMuted)
			{
				this.CurrentObject.Player.Volume = volume;
			}
			this.lastVolume = volume;
		}

		private void ToggleZoom()
		{
			// If value increased, double, else halve
			if (this.ZoomNumeric.Value > this.oldZoomValue)
			{
				this.ZoomNumeric.Value = Math.Min(this.ZoomNumeric.Maximum, this.oldZoomValue * 2);
			}
			else
			{
				this.ZoomNumeric.Value = Math.Max(this.ZoomNumeric.Minimum, this.oldZoomValue / 2);
			}

			this.oldZoomValue = (int) this.ZoomNumeric.Value;
		}

		private void UpdateTrackList()
		{
			if (this.TrackList.InvokeRequired)
			{
				this.TrackList.Invoke((MethodInvoker) (() => {
					this.TrackList.DataSource = null;
					this.TrackList.DataSource = this.Tracks;
					this.TrackList.DisplayMember = "Name";
				}));
			}
			else
			{
				this.TrackList.DataSource = null;
				this.TrackList.DataSource = this.Tracks;
				this.TrackList.DisplayMember = "Name";
			}
		}

		private void UpdateTrackInfo()
		{
			if (this.CurrentObject == null)
			{
				return;
			}

			// Korrekte Länge berechnen (in Samples)
			int totalSamples = this.CurrentObject.Data.Length / this.CurrentObject.Channels;
			this.OffsetScroll.Maximum = Math.Max(0, totalSamples - (this.WavePBox.Width * (int) this.ZoomNumeric.Value));
			this.OffsetScroll.LargeChange = this.WavePBox.Width;
			this.OffsetScroll.SmallChange = this.WavePBox.Width / 10;

			// Update MetaLabel
			this.MetaLabel.Text = $"{totalSamples / this.CurrentObject.Samplerate:0.00}s | " +
								  $"{this.CurrentObject.Samplerate}Hz | " +
								  $"{this.CurrentObject.Bitdepth}bit | " +
								  $"{this.CurrentObject.Channels}ch";
		}

		private void UpdateWaveform()
		{
			if (this.WavePBox.InvokeRequired)
			{
				this.WavePBox.Invoke((MethodInvoker) (() => {
					this.WavePBox.Image = this.CurrentWaveform;
				}));
			}
			else
			{
				this.WavePBox.Image = this.CurrentWaveform;
			}
		}

		private void UpdatePlaybackPosition()
		{
			while (this.isPlaying && !this.playbackCancellation.IsCancellationRequested)
			{
				if (this.CurrentObject == null)
				{
					break;
				}

				double currentTime = this.CurrentObject.CurrentTime;
				int currentSample = (int) (currentTime * this.CurrentObject.Samplerate);

				this.InvokeIfRequired(() =>
				{
					// Update TimeText
					this.TimeText.Text = currentTime.ToString("0.00");

					// Update Scroll Position
					int visibleRange = this.WavePBox.Width * (int) this.ZoomNumeric.Value;
					int targetScrollPos = currentSample - (visibleRange / 2);

					if (targetScrollPos != this.OffsetScroll.Value)
					{
						this.OffsetScroll.Value = Math.Max(0, Math.Min(targetScrollPos, this.OffsetScroll.Maximum));
					}

					// Force waveform redraw
					this.WavePBox.Image = this.CurrentWaveform;
				});

				Thread.Sleep(30); // Faster updates for smoother movement
			}
		}

		private void TogglePlayback()
		{
			if (this.CurrentObject == null)
			{
				return;
			}

			if (this.isPlaying)
			{
				this.playbackCancellation.Cancel();
				this.CurrentObject.Stop();
				this.UpdateButtonState(false);
				this.isPlaying = false;
			}
			else
			{
				this.playbackCancellation = new CancellationTokenSource();

				// Lautstärke aus VolumeScroll lesen (invertiert: 100 → 0.0f, 0 → 1.0f)
				float initialVolume = (100 - this.VolumeScroll.Value) / 100f;

				// Play mit initialer Lautstärke aufrufen
				this.CurrentObject.Play(this.playbackCancellation.Token, () =>
				{
					this.UpdateButtonState(false);
					this.isPlaying = false;
				}, initialVolume);

				this.UpdateButtonState(true);
				this.isPlaying = true;
				Task.Run(this.UpdatePlaybackPosition, this.playbackCancellation.Token);
			}
		}

		private void InvokeIfRequired(Action action)
		{
			if (this.PlayButton.InvokeRequired)
			{
				this.PlayButton.Invoke(action);
			}
			else
			{
				action();
			}
		}

		private void UpdateButtonState(bool playing)
		{
			if (this.PlayButton.InvokeRequired)
			{
				this.PlayButton.Invoke((MethodInvoker) (() => {
					this.PlayButton.Text = playing ? "■" : "▶";
				}));
			}
			else
			{
				this.PlayButton.Text = playing ? "■" : "▶";
			}
		}

		private void ImportResourcesAudio(bool silent = false)
		{
			try
			{
				string dirPath = Path.Combine(this.Repopath, "Resources", "Audios");
				if (!Directory.Exists(dirPath))
				{
					return;
				}

				string[] files = Directory.GetFiles(dirPath, "*.*", SearchOption.AllDirectories)
					.Where(s => s.EndsWith(".mp3") || s.EndsWith(".wav"))
					.ToArray();

				foreach (string file in files)
				{
					this.AddTrack(file, silent);
				}
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					this.Log("Error loading resources", ex.Message, 1);
				}
			}
		}
	}

	public class AudioObject
	{
		// ----- ----- ----- ATTRIBUTES ----- ----- ----- \\
		public string Filepath { get; set; }
		public string Name { get; set; }
		public float[] Data { get; set; } = [];
		public int Samplerate { get; set; } = -1;
		public int Bitdepth { get; set; } = -1;
		public int Channels { get; set; } = -1;
		public long Length { get; set; } = -1;

		public IntPtr[] Pointers { get; set; } = [];
		public int ChunkSize { get; private set; } = 0;
		public int OverlapSize { get; private set; } = 0;

		public WaveOutEvent Player { get; set; } = new WaveOutEvent();

		// ----- ----- ----- PROPERTIES ----- ----- ----- \\
		public long Position
		{
			get
			{
				return this.Player == null || this.Player.PlaybackState != PlaybackState.Playing
					?  0
					: this.Player.GetPosition() / (this.Channels * (this.Bitdepth / 8));
			}
		}

		public double CurrentTime
		{
			get
			{
				return this.Samplerate <= 0 ?  0 : (double) this.Position / this.Samplerate;
			}
		}


		public bool OnHost => this.Data.Length > 0 && this.Pointers.Length == 0;
		public bool OnDevice => this.Data.Length == 0 && this.Pointers.Length != 0;


		// ----- ----- ----- CONSTRUCTOR ----- ----- ----- \\
		public AudioObject(string filepath)
		{
			this.Filepath = filepath;
			this.Name = Path.GetFileNameWithoutExtension(filepath);
			this.LoadAudioFile();
		}

		public void LoadAudioFile()
		{
			if (string.IsNullOrEmpty(this.Filepath))
			{
				throw new FileNotFoundException("File path is empty");
			}

			using AudioFileReader reader = new(this.Filepath);
			this.Samplerate = reader.WaveFormat.SampleRate;
			this.Bitdepth = reader.WaveFormat.BitsPerSample;
			this.Channels = reader.WaveFormat.Channels;
			this.Length = reader.Length; // Length in bytes

			// Calculate number of samples
			long numSamples = reader.Length / (reader.WaveFormat.BitsPerSample / 8);
			this.Data = new float[numSamples];

			int read = reader.Read(this.Data, 0, (int) numSamples);
			if (read != numSamples)
			{
				float[] resizedData = new float[read];
				Array.Copy(this.Data, resizedData, read);
				this.Data = resizedData;
			}
		}

		public byte[] GetBytes()
		{
			int bytesPerSample = this.Bitdepth / 8;
			byte[] bytes = new byte[this.Data.Length * bytesPerSample];

			Parallel.For(0, this.Data.Length, i =>
			{
				switch (this.Bitdepth)
				{
					case 8:
						bytes[i] = (byte) (this.Data[i] * 127);
						break;
					case 16:
						short sample16 = (short) (this.Data[i] * short.MaxValue);
						Buffer.BlockCopy(BitConverter.GetBytes(sample16), 0, bytes, i * bytesPerSample, bytesPerSample);
						break;
					case 24:
						int sample24 = (int) (this.Data[i] * 8388607);
						Buffer.BlockCopy(BitConverter.GetBytes(sample24), 0, bytes, i * bytesPerSample, 3);
						break;
					case 32:
						Buffer.BlockCopy(BitConverter.GetBytes(this.Data[i]), 0, bytes, i * bytesPerSample, bytesPerSample);
						break;
				}
			});

			return bytes;
		}

		public List<float[]> GetChunksOld(int size = 2048, float overlap = 0.5f)
		{
			if (this.Data == null || this.Data.Length == 0)
			{
				return [];
			}

			if (size <= 0 || overlap < 0 || overlap >= 1)
			{
				return [];
			}

			this.ChunkSize = size;
			this.OverlapSize = (int) (size * overlap);
			int step = size - this.OverlapSize;
			int numChunks = (this.Data.Length - size) / step + 1;

			List<float[]> chunks = new List<float[]>(numChunks);

			for (int i = 0; i < numChunks; i++)
			{
				float[] chunk = new float[size];
				int sourceOffset = i * step;
				Array.Copy(this.Data, sourceOffset, chunk, 0, size);
				chunks.Add(chunk);
			}

			return chunks;
		}

		public List<float[]> GetChunks(int size = 2048, float overlap = 0.5f)
		{
			if (this.Data == null || this.Data.Length == 0)
			{
				return [];
			}

			if (size <= 0 || overlap < 0 || overlap >= 1)
			{
				return [];
			}

			this.ChunkSize = size;
			this.OverlapSize = (int) (size * overlap);
			int step = size - this.OverlapSize;
			int numChunks = (this.Data.Length - size) / step + 1;

			// Parallel vorbereiten
			float[][] chunks = new float[numChunks][];

			Parallel.For(0, numChunks, i =>
			{
				int sourceOffset = i * step;
				float[] chunk = new float[size];
				Array.Copy(this.Data, sourceOffset, chunk, 0, size);
				chunks[i] = chunk;
			});

			// In List umwandeln
			return chunks.ToList();
		}


		public void AggregateChunks(List<float[]> chunks)
		{
			if (chunks == null || chunks.Count == 0)
			{
				return;
			}

			int size = this.ChunkSize;
			int step = size - this.OverlapSize;
			int outputLength = (chunks.Count - 1) * step + size;

			// Zielpuffer vorbereiten
			float[] output = new float[outputLength];
			float[] weightSum = new float[outputLength];

			// Parallel lokal puffern, dann atomar addieren
			int processorCount = Environment.ProcessorCount;

			Parallel.For(0, chunks.Count, () =>
				(new float[outputLength], new float[outputLength]),

				(i, state, local) =>
				{
					(Single[] localData, Single[] localWeight) = local;
					float[] chunk = chunks[i];
					int offset = i * step;

					for (int j = 0; j < size; j++)
					{
						int idx = offset + j;
						localData[idx] += chunk[j];
						localWeight[idx] += 1f;
					}
					return (localData, localWeight);
				},

				local =>
				{
					(Single[] localData, Single[] localWeight) = local;
					lock (output) // kurzes Lock für zusammenführung
					{
						for (int i = 0; i < outputLength; i++)
						{
							output[i] += localData[i];
							weightSum[i] += localWeight[i];
						}
					}
				}
			);

			// Normalisierung
			Parallel.For(0, outputLength, i =>
			{
				if (weightSum[i] > 0f)
				{
					output[i] /= weightSum[i];
				}
			});

			// Setze das Ergebnis
			this.Data = output;
		}



		public void Play(CancellationToken cancellationToken, Action? onPlaybackStopped = null, float initialVolume = 1.0f)
		{
			if (this.Data == null || this.Data.Length == 0)
			{
				return;
			}

			this.Player = new WaveOutEvent
			{
				Volume = initialVolume
			};

			byte[] bytes = this.GetBytes();
			WaveFormat waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(this.Samplerate, this.Channels);
			RawSourceWaveStream stream = new(new MemoryStream(bytes), waveFormat);

			this.Player.PlaybackStopped += (s, e) =>
			{
				onPlaybackStopped?.Invoke();
				stream.Dispose();
				this.Player.Dispose();
			};

			this.Player.Init(stream);
			this.Player.Play();

			// Überprüfe regelmäßig auf Abbruch
			Task.Run(() =>
			{
				while (this.Player.PlaybackState == PlaybackState.Playing)
				{
					if (cancellationToken.IsCancellationRequested)
					{
						this.Player.Stop();
						break;
					}
					Thread.Sleep(100);
				}
			}, cancellationToken);
		}

		public void Stop()
		{
			this.Player.Stop();
		}

		public double GetCurrentTime()
		{
			if (this.Player.PlaybackState != PlaybackState.Playing)
			{
				return 0;
			}

			// Korrekte Berechnung der aktuellen Zeit
			return (double) this.Player.GetPosition() /
				   (this.Samplerate * this.Channels * (this.Bitdepth / 8));
		}

		public Bitmap DrawWaveformParallel(PictureBox pictureBox, int samplesPerPixel = 1024, long offset = 0, Color? graphColor = null)
		{
			graphColor ??= Color.BlueViolet;
			Bitmap bitmap = new(pictureBox.Width, pictureBox.Height);

			using (Graphics graphics = Graphics.FromImage(bitmap))
			using (Pen wavePen = new(graphColor.Value, 1f))
			{
				graphics.Clear(pictureBox.BackColor);

				if (this.Data == null || this.Data.Length == 0)
				{
					return bitmap;
				}

				int width = pictureBox.Width;
				int height = pictureBox.Height;
				float[] data = this.Data;
				int channels = this.Channels;

				// Draw waveform
				Parallel.For(0, width, x =>
				{
					long sampleIndex = offset + (x * samplesPerPixel);
					if (sampleIndex * channels >= data.Length - channels)
					{
						return;
					}

					if (channels == 2)
					{
						float left = data[sampleIndex * 2];
						float right = data[sampleIndex * 2 + 1];

						int leftY = (int) (height * 0.25f - left * height * 0.2f);
						int rightY = (int) (height * 0.75f - right * height * 0.2f);

						lock (graphics)
						{
							graphics.DrawLine(wavePen, x, height * 0.25f, x, leftY);
							graphics.DrawLine(wavePen, x, height * 0.75f, x, rightY);
						}
					}
					else
					{
						float sample = data[sampleIndex];
						int y = (int) (height * 0.5f - sample * height * 0.4f);

						lock (graphics)
						{
							graphics.DrawLine(wavePen, x, height * 0.5f, x, y);
						}
					}
				});

				// Draw playhead if playing
				if (this.Player.PlaybackState == PlaybackState.Playing)
				{
					int playheadX = (int) ((this.Position - offset) / samplesPerPixel);

					if (playheadX >= 0 && playheadX < width)
					{
						using (Pen playheadPen = new(Color.Red, 3))
						{
							graphics.DrawLine(playheadPen, playheadX, 0, playheadX, height);
							graphics.FillRectangle(Brushes.Red, playheadX - 2, 0, 5, 10);
							graphics.FillRectangle(Brushes.Red, playheadX - 2, height - 10, 5, 10);
						}
					}
				}
			}

			return bitmap;
		}

		public string? Export()
		{
			using SaveFileDialog sfd = new();
			sfd.Title = "Export audio file";
			sfd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
			sfd.Filter = "Wave files (*.wav)|*.wav|MP3 files (*.mp3)|*.mp3";
			sfd.OverwritePrompt = true;

			if (sfd.ShowDialog() == DialogResult.OK)
			{
				byte[] bytes = this.GetBytes();
				WaveFormat waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(this.Samplerate, this.Channels);

				using (RawSourceWaveStream stream = new(new MemoryStream(bytes), waveFormat))
				using (FileStream fileStream = new(sfd.FileName, FileMode.Create))
				{
					WaveFileWriter.WriteWavFileToStream(fileStream, stream);
				}

				return sfd.FileName;
			}
			return null;
		}

		public void Reload()
		{
			// Null pointer
			this.Pointers = [];
		
			this.LoadAudioFile();
		}

		public void Normalize(float maxAmplitude = 1.0f)
		{
			if (this.Data == null || this.Data.Length == 0)
			{
				return;
			}

			// Schritt 1: Maximalwert (Betrag) ermitteln – parallel
			float globalMax = 0f;
			object lockObj = new object();

			Parallel.For(0, this.Data.Length, () => 0f, (i, _, localMax) =>
			{
				float abs = Math.Abs(this.Data[i]);
				return abs > localMax ? abs : localMax;
			},
			localMax =>
			{
				lock (lockObj)
				{
					if (localMax > globalMax)
					{
						globalMax = localMax;
					}
				}
			});

			// Kein Normalisieren nötig, wenn max = 0
			if (globalMax == 0f)
			{
				return;
			}

			// Schritt 2: Daten parallel skalieren
			float scale = maxAmplitude / globalMax;

			Parallel.For(0, this.Data.Length, i =>
			{
				this.Data[i] *= scale;
			});
		}

	}
}