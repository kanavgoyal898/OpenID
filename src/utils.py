import scipy
import librosa
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 11025         # Sample rate for audio files (~11 kHz)
WINDOW_SIZE = 1024          # Window size for framing audio
OVERLAP = 0.5               # Overlap between frames
FREQUENCY_START = 20        # 20 Hz
FREQUENCY_END = 5120        # 5.12 kHz
KERNEL_SIZE = 3             # Kernel size for peak picking
THRESHOLD_INTENSITY = -40   # Minimum intensity (in dB) to consider a peak
K = 7                       # Number of peaks to select per time frame in each band
TIME_INTERVAL = 2           # Time interval (in seconds) for pairing peaks

def load_song(file_path, sample_rate=SAMPLE_RATE):
    song, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return song, sr

def frame_audio(audio, window_size=WINDOW_SIZE, overlap=OVERLAP):
    hop_length = int(window_size * (1 - overlap))
    frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_length).T
    return frames

def compute_stft(signal, n_fft=WINDOW_SIZE, hop_length=int(WINDOW_SIZE*(1 - OVERLAP))):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window="hann", center=False)
    return stft

def plot_spectrogram(stft, sample_rate=SAMPLE_RATE, n_fft=WINDOW_SIZE, freq_start=FREQUENCY_START, freq_end=FREQUENCY_END, verbose=False):
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    
    # Filter frequencies between 20 Hz and 5.12 kHz
    mask = (freqs >= freq_start) & (freqs <= freq_end)      
    db = db[mask, :]
    
    if verbose:
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(db, sr=sample_rate, x_axis="time", y_axis="linear", cmap="magma")
        plt.colorbar(format="%+3.0f dB")
        plt.title("Spectrogram (dB)")
        plt.tight_layout()
        plt.show()

    return db

def select_peaks(spectrogram, sample_rate=SAMPLE_RATE, n_fft=WINDOW_SIZE, freq_start=FREQUENCY_START, freq_end=FREQUENCY_END, kernel_size=KERNEL_SIZE, threshold_intensity=THRESHOLD_INTENSITY, k=K, verbose=False):
    bands = int(np.log2(freq_end / freq_start))

    band_splits = [freq_start]
    for i in range(bands):
        band_splits.append(band_splits[-1] * 2)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    freqs = freqs[(freqs >= freq_start) & (freqs <= freq_end)]

    peaks = []

    for start_freq, end_freq in zip(band_splits[:-1], band_splits[1:]):
        band_mask = (freqs >= start_freq) & (freqs < end_freq)
        
        band_indices = np.where(band_mask)[0]
        if len(band_indices) == 0:
            continue

        band_spectrogram = spectrogram[band_indices, :]

        maximas = scipy.ndimage.maximum_filter(band_spectrogram, size=(kernel_size, kernel_size), mode="constant", cval=-np.inf) == band_spectrogram
        band_peaks = np.argwhere(maximas)

        for time_idx in np.unique(band_peaks[:, 1]):
            time_peaks = band_peaks[band_peaks[:, 1] == time_idx]
            time_peaks = sorted(time_peaks, key=lambda x: band_spectrogram[x[0], x[1]], reverse=True)
            for time_peak in time_peaks[:k]:
                freq_idx, t_idx = time_peak
                magnitude = band_spectrogram[freq_idx, t_idx]
                if magnitude >= threshold_intensity:
                    peaks.append((band_indices[freq_idx], t_idx, magnitude))

    if verbose:
        hop_length = int(WINDOW_SIZE * (1 - OVERLAP))
        peak_times = [peak_time * hop_length / sample_rate for (freq_idx, peak_time, _) in peaks]
        peak_freqs = [freqs[freq_idx] for freq_idx, _, _ in peaks]

        plt.figure(figsize=(12, 6))
        plt.scatter(peak_times, peak_freqs, s=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Anchor Points")
        plt.show()

    return peaks

def encode(f1, f2, dt):
    # 32-bit hash: [f1: 12 bits][f2: 12 bits][dt: 8 bits]
    return (f1 & 0xFFF) << 20 | (f2 & 0xFFF) << 8 | (dt & 0xFF)

def generate_hashes(peaks, hop_length=int(WINDOW_SIZE * (1 - OVERLAP)), time_interval=TIME_INTERVAL, sample_rate=SAMPLE_RATE):
    hashes = []

    peaks = sorted(peaks, key=lambda x: x[1])

    for ix, (f1_idx, t1_idx, _) in enumerate(peaks):
        t1 = t1_idx * hop_length / sample_rate
         
        for jx, (f2_idx, t2_idx, _) in enumerate(peaks[ix + 1:]):
            t2 = t2_idx * hop_length / sample_rate

            dt = t2 - t1
            if 0 < dt <= time_interval:
                dt = int(dt * 100)
                hash_value = encode(f1_idx, f2_idx, dt)
                hashes.append((int(hash_value), int(t1 * 1000)))
            else:
                break

    return hashes
