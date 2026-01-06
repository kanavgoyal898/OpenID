from utils import *

import random

PATH = input("Enter path to audio file: ")
hashes = generate_hashes(select_peaks(plot_spectrogram(compute_stft(load_song(PATH)[0]))))

n = len(hashes)
print(f"Generated {n:,} hashes.")
idx = random.randint(0, n - 1)
print(f"Sample hash {idx:,}: {hashes[idx]}")
