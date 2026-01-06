# OpenID: Audio Fingerprinting & Recognition System

A powerful Python-based audio recognition system that identifies songs from recorded audio by analyzing their unique acoustic fingerprints. Using advanced signal processing techniques including Short-Time Fourier Transform (STFT), spectral peak detection, and combinatorial hash generation, this implementation creates a robust fingerprinting database that can accurately match audio samples even in noisy environments.

<div style="text-align: center;">
  <img src="./image.png" alt="OpenID" style="width: 100%;">
</div>

The system works by converting audio signals into time-frequency representations (spectrograms), identifying prominent spectral peaks that represent the most distinctive features of a song, and encoding these peaks into compact 32-bit hash values. These fingerprints are stored in an efficient SQLite database, enabling fast lookups and matching against thousands of songs in milliseconds.

OpenID uses audio fingerprinting technology to identify songs from short recordings. The system:
1. Converts audio into spectrograms
2. Extracts distinctive acoustic features (peaks)
3. Generates compact hash fingerprints
4. Matches recorded audio against a database of known songs

## Features

- **Audio Recording**: Capture audio directly from your microphone
- **Fingerprint Generation**: Create unique acoustic signatures for songs
- **Fast Matching**: Efficiently query large song databases
- **Noise Tolerant**: Works even with background noise and audio distortion
- **Frequency Band Analysis**: Logarithmic frequency bands for robust peak detection

## Installation

### Prerequisites

- Python 3.7 or higher
- A microphone (for audio recognition)
- MP3 audio files (for building the database)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kanavgoyal898/OpenID
cd OpenID
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create the necessary directories:
```bash
mkdir songs
```

## Usage

### Building the Song Database

1. Add your MP3 files to the `songs/` directory
2. Run the program and select 'N' when prompted:
```bash
python src/main.py
```
3. The system will process all songs and store their fingerprints in the database

### Recognizing Audio

1. Run the program and select 'Y' when prompted:
```bash
python src/main.py
```
2. Wait for the countdown (default: 3 seconds)
3. Play the song you want to identify near your microphone
4. The system will record for 10 seconds and attempt to match the audio

## Configuration

You can modify the following parameters in `src/utils.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 11025 Hz | Audio sampling rate |
| `WINDOW_SIZE` | 1024 | FFT window size |
| `OVERLAP` | 0.5 | Frame overlap ratio |
| `FREQUENCY_START` | 20 Hz | Minimum frequency analyzed |
| `FREQUENCY_END` | 5120 Hz | Maximum frequency analyzed |
| `TIME_INTERVAL` | 2 seconds | Maximum time gap for peak pairing |
| `FAN_OUT` | 7 | Number of pairs per anchor point |
| `DURATION` | 10 seconds | Recording duration |
| `COUNTDOWN` | 3 seconds | Countdown before recording |

## How It Works

### 1. Audio Processing
- Audio is loaded and converted to mono at 11.025 kHz
- Short-Time Fourier Transform (STFT) creates a spectrogram
- Frequency range is filtered to 20 Hz - 5.12 kHz

### 2. Peak Detection
- Spectrogram is divided into logarithmic frequency bands
- Local maxima are detected using maximum filters
- Top K peaks per time frame in each band are selected

### 3. Fingerprint Generation
- Peaks are paired within a time window (anchor-target pairs)
- Each pair generates a 32-bit hash: `[freq1: 12 bits][freq2: 12 bits][Δt: 8 bits]`
- Hashes are stored with their time offsets

### 4. Song Recognition
- Query audio generates fingerprints using the same process
- Fingerprints are matched against the database
- Time-offset voting determines the best match

## Project Structure

```
OpenID/
├── src/
│   ├── main.py
│   ├── utils.py
│   └── schema.sql
├── songs/
├── fingerprints.db
├── requirements.txt
└── README.md 
```

## Technical Details

### Algorithm Complexity
- **Storage**: O(n×m) where n = songs, m = avg hashes per song
- **Query**: O(h×k) where h = query hashes, k = avg matches per hash
- **Recognition**: O(m) for voting algorithm

### Performance Tips
- More songs require more processing time
- Higher sample rates increase accuracy but slow processing
- Adjust `FAN_OUT` to balance accuracy vs. database size
