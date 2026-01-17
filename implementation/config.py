# PARAMETERS AND CONFIGURATIONS

from dataclasses import dataclass

# SEED
SEED = 42

# Paths
DATASET_PATH = r"..\datasets\MUSDB18-7"

@dataclass
class AudioConfig:
    sample_rate = 8192
    n_fft = 1024
    hop_length = 768
    num_frames = 64 # TODO : revert this back to 128