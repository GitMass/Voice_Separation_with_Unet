import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset
import musdb

from .config import SEED, DATASET_PATH, AudioConfig

random.seed(SEED)




def data_feeder(mus_tracks, audio_config, num_samples) : 

    num_frames = audio_config.num_frames
    hop_length = audio_config.hop_length
    n_fft = audio_config.n_fft
    sample_rate = audio_config.sample_rate

    # sample duration
    desired_duration = num_frames * (hop_length/sample_rate)

    for _ in range(num_samples) :

        yield_next = True
        while yield_next : 
            
            # choose track 
            track = random.choice(mus_tracks)

            # verify duration
            if track.duration < desired_duration :
                print(f"track skipped : too short")
                continue

            # adjust load duration 
            track.chunk_duration = desired_duration

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

            # Frames: Découpe pour s'assurer qu'on a exactement 128 frames (parfois librosa ajoute 1 frame)
            # Freq: On passe de 513 à 512 (on retire la dernière bin Nyquist)
            spec_mix = spec_mix[:-1, :num_frames]
            spec_vocals = spec_vocals[:-1, :num_frames]

            # Ajout de la dimension de canal (pour faire 1x512x128)
            spec_mix = spec_mix[np.newaxis, :, :]       # (1, 512, 128)
            spec_vocals = spec_vocals[np.newaxis, :, :] # (1, 512, 128)

            # Normalisation [0, 1] (we use the mix max)
            max_mix = np.max(spec_mix) + 1e-8
            spec_mix = spec_mix / max_mix
            spec_vocals = spec_vocals / max_mix
            
            # convert to tensor
            x = torch.tensor(spec_mix, dtype=torch.float32)
            y = torch.tensor(spec_mix, dtype=torch.float32)

            # Yield of the example
            yield (x, y)
            yield_next = False
        




class SingingVoiceDataset(IterableDataset) :

    def __init__(self, split='train'):
        
        # init database
        self.mus = musdb.DB(root=DATASET_PATH, download=False, subsets='train')
        self.audio_config = AudioConfig()

        # train validation split (80%)
        all_tracks = self.mus.tracks
        random.shuffle(all_tracks)
        split_idx = int(0.8 * len(all_tracks))
        if split == 'train' : 
            self.tracks = all_tracks[:split_idx]
        elif split == 'validation' :
            self.tracks = all_tracks[split_idx:]

        # Calculate sample_duration
        self.sample_duration = self.audio_config.num_frames * (self.audio_config.hop_length / self.audio_config.sample_rate)

        # Calculate total duration and num_samples_per_epoch (the variable for ~1 full pass)
        self.total_duration = sum(track.duration for track in self.tracks)
        self.num_samples_per_epoch = int(self.total_duration / self.sample_duration) if self.sample_duration > 0 else 0
        print(f"num_samples_per_epoch for {split} : {self.num_samples_per_epoch}")

    def __iter__(self):
        # We call our previously defined datafeeder
        return data_feeder(self.tracks, self.audio_config, self.num_samples_per_epoch)
        



def display_spectrogram(spec, title="Spectrogram"):
    """
    Displays a spectrogram with proper axes and scaling.
    
    Args:
        spec (np.array): Input spectrogram of shape (Freq, Time) or (Freq, Time, 1)
        title (str): Title of the plot
        sr (int): Sampling rate (default 8192Hz per Article)
        hop_length (int): Hop length (default 768 per Article)
        to_db (bool): If True, converts magnitude to decibels for display
    """
    plt.figure(figsize=(10, 4))
    
    # 1. Handle Dimensions: Remove the channel dimension (512, 12, 1) -> (512, 12)
    if spec.ndim == 3:
        spec = spec.squeeze()

    # 2. Display
    plt.imshow(spec, origin='lower', aspect='auto', vmin=0, vmax=1)
    
    plt.colorbar()
    plt.title(title)
    plt.xlabel("window")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()