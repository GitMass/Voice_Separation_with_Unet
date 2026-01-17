# PARAMETERS AND CONFIGURATIONS

from dataclasses import dataclass

# SEED
SEED = 42

# Paths
DATASET_PATH = r"datasets\MUSDB18-full"

@dataclass
class AudioConfig:
    sample_rate = 8192
    n_fft = 1024
    hop_length = 768
    num_frames = 128