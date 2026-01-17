import numpy as np
import random
import librosa

from config import SEED

random.seed(SEED)




def data_feeder(mus_tracks, audio_config , batch_size) : 

    num_frames = audio_config.num_frames
    hop_length = audio_config.hop_length
    n_fft = audio_config.n_fft
    sample_rate = audio_config.sample_rate

    # sample duration
    desired_duration = num_frames * (hop_length/sample_rate)

    # init batch
    batch_x = []
    batch_y = []

    while True : 
        print("fetching new track")
        
        # choose track 
        track = random.choice(mus_tracks)

        # adjust load duration 
        track.chunk_duration = desired_duration

        # verify duration
        if track.duration < desired_duration :
            print(f"track skipped : too short")
            continue

        # choose a starting point
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)

        # get audio (Mix et Voix) (mean(axis=1) pour convertir stéréo -> mono si nécessaire)
        audio_mix = track.audio.mean(axis=1) 
        audio_vocals = track.targets['vocals'].audio.mean(axis=1)

        # Sous-échantillonnage
        audio_mix = librosa.resample(audio_mix, orig_sr=track.rate, target_sr=sample_rate)
        audio_vocals = librosa.resample(audio_vocals, orig_sr=track.rate, target_sr=sample_rate)

        # Calcul du Spectrogramme (STFT) (On ne garde que la Magnitude (abs))
        spec_mix = np.abs(librosa.stft(audio_mix, n_fft=n_fft, hop_length=hop_length))
        spec_vocals = np.abs(librosa.stft(audio_vocals, n_fft=n_fft, hop_length=hop_length))

        # Découpe pour s'assurer qu'on a exactement 128 frames (parfois librosa ajoute 1 frame)
        spec_mix = spec_mix[:, :num_frames]
        spec_vocals = spec_vocals[:, :num_frames]

        # Ajout de la dimension de canal (pour faire 512x128x1)
        spec_mix = spec_mix[..., np.newaxis]
        spec_vocals = spec_vocals[..., np.newaxis]

        # Normalisation [0, 1] (we use the mix max)
        max_mix = np.max(spec_mix) + 1e-8
        spec_mix = spec_mix / max_mix
        spec_vocals = spec_vocals / max_mix
        
        # Append results
        batch_x.append(spec_mix)
        batch_y.append(spec_vocals)

        # Yield du Batch
        if len(batch_x) == batch_size:
            # Conversion en array numpy: (batch_size, 513, 128, 1)
            # Note: 513 bins car n_fft//2 + 1. L'article dit 512, ils ignorent souvent la fréquence Nyquist (la dernière bin)
            yield np.array(batch_x)[:, :-1, :, :], np.array(batch_y)[:, :-1, :, :]
            
            # Reset
            batch_x, batch_y = [], []

