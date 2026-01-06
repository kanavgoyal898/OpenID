from utils import *

user = input("Audio Recognition? [Y/N]: ")
if user.lower() == 'y':
    with record_audio(verbose=True) as path:
        hashes = generate_hashes(select_peaks(plot_spectrogram(compute_stft(load_song(path)[0]))))
        print(f"Generated {len(hashes):,} hashes from recorded audio.")
        
        matches = query_fingerprints(hashes)
        song = recognize_song(matches)
    
        if song:
            print(f"Recognized song: {song}")
        else:
            print("No matching song found.")
else:
    initialize_database()
    n = store_fingerprints(verbose=True)
    print(f"Processed {n} songs.")
