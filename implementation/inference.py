import torch
import librosa
import numpy as np
import soundfile as sf
import argparse
import matplotlib.pyplot as plt
import os

from .model import UnetSeparator
from .config import AudioConfig, CHECKPOINT_PATH, INFERENCE_RESULTS_PATH

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def separate_audio(audio_path, model):

    """
    Sépare un fichier audio en Vocals et Accompagnement.
    Gère le padding pour le U-Net et la reconstruction par phase.
    """

    # load cnfig
    config = AudioConfig()

    # init dir
    os.makedirs(INFERENCE_RESULTS_PATH, exist_ok=True)
    filename = os.path.basename(audio_path).split('.')[0]

    print(f"--- Traitement de : {filename} ---")

    # Chargement Audio (mono)
    audio_mono, sr = librosa.load(audio_path, sr=config.sample_rate)

    # STFT
    stft_mono = librosa.stft(audio_mono, n_fft=config.n_fft, hop_length=config.hop_length)
    mag_mono_513= np.abs(stft_mono)
    phase_mono_513 = np.angle(stft_mono)

    # crop to 512 for model compatibility
    mag_mono = mag_mono_513[:-1, :] # (512, Time)

    # 3. PADDING (CRITIQUE pour U-Net)
    # Le U-Net a 6 couches de downsampling (2^6 = 64). 
    # La dimension temporelle DOIT être un multiple de 64.
    num_frames = mag_mono.shape[1]
    pad_len = 64 - (num_frames % 64)
    if pad_len == 64: 
        pad_len = 0
        
    # On ajoute du padding (zéros) seulement à la fin de l'axe temporel
    # shape: (Freq, Time) -> (Freq, Time + pad_len)
    if pad_len > 0:
        mag_mono_padded = np.pad(mag_mono, ((0, 0), (0, pad_len)))
    else:
        mag_mono_padded = mag_mono
    
    # On normalise la magnitude
    max_val = np.max(mag_mono_padded) + 1e-8
    mag_mono_padded_normalized = mag_mono_padded / max_val
    
    # Conversion Tensor
    tensor_input = torch.tensor(mag_mono_padded_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Inférence (Forward Pass)
    print("Calcul du masque (foorward) ...")
    model.eval()
    with torch.no_grad():

        # Le modèle retourne : mask * input. (vocals)
        pred_mag_vocals_padded_normalised = model(tensor_input)
        pred_mag_vocals_padded_normalised = pred_mag_vocals_padded_normalised.squeeze().cpu().numpy() # [Freq, Time_Padded]
                
        # On dé-normalise
        pred_mag_padded_vocals = pred_mag_vocals_padded_normalised * max_val

    # 7. Retrait du Padding (Un-pad)
    if pad_len > 0:
        pred_mag_vocals = pred_mag_padded_vocals[:, :-pad_len]
    else:
        pred_mag_vocals = pred_mag_padded_vocals

    # Création du Masque "Souple" (Soft Mask) : Mask = |Vocals_Predicted| / (|Mix| + epsilon)
    # C'est souvent plus propre que d'utiliser la sortie brute, car ça conserve l'énergie totale
    mask_vocals = pred_mag_vocals / (mag_mono + 1e-8)
    mask_vocals = np.clip(mask_vocals, 0, 1) # Garder entre 0 et 1
    mask_vocals_513 = np.pad(mask_vocals, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    
    mask_instru_513 = 1.0 - mask_vocals_513

    # Reconstruction Audio
    print("Reconstruction audio...")
    
    # Vocals = Mix_Stereo * Mask_Vocals * Phase_Originale
    predicted_stft_vocals = (mag_mono_513 * mask_vocals_513) * np.exp(1j * phase_mono_513)
    predicted_stft_instru = (mag_mono_513 * mask_instru_513) * np.exp(1j * phase_mono_513)

    # iSTFT
    predicted_audio_vocals = []
    predicted_audio_instru = []
    
    predicted_audio_vocals.append(librosa.istft(predicted_stft_vocals, hop_length=config.hop_length))
    predicted_audio_instru.append(librosa.istft(predicted_stft_instru, hop_length=config.hop_length))
        
    predicted_audio_vocals = np.array(predicted_audio_vocals).T # (Samples, 2)
    predicted_audio_instru = np.array(predicted_audio_instru).T

    # Sauvegarde
    original_path = os.path.join(INFERENCE_RESULTS_PATH, f"{filename}_original_mix.wav") # mix
    vocal_path = os.path.join(INFERENCE_RESULTS_PATH, f"{filename}_predicted_vocals.wav")
    instru_path = os.path.join(INFERENCE_RESULTS_PATH, f"{filename}_predicted_instru.wav")
    

    sf.write(original_path, audio_mono, sr)
    sf.write(vocal_path, predicted_audio_vocals, sr)
    sf.write(instru_path, predicted_audio_instru, sr)

    # plots

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Spectrogramme Entrée (Mix)")
    plt.imshow(np.log1p(mag_mono), aspect='auto', origin='lower')

    ax2 = plt.subplot(1, 3, 2)
    plt.title("Masque Prédit par le Modèle")
    im = plt.imshow(mask_vocals, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2)

    plt.subplot(1, 3, 3)
    plt.title("Voix Isolée (Mix * Masque)")
    plt.imshow(np.log1p(mag_mono * mask_vocals), aspect='auto', origin='lower')

    plt.show()

    
    print(f"Terminé ! Fichiers sauvegardés dans {INFERENCE_RESULTS_PATH}")




if __name__ == "__main__":

    # Charger le modèle
    model = UnetSeparator().to(DEVICE)

    # Load weights
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Poids chargés.")
    else:
        print("ERREUR : Checkpoint introuvable. Entraînez le modèle d'abord.")
        exit()

    # Exemple d'utilisation
    # Vous pouvez changer le chemin ici vers un fichier wav de test
    test_file = "datasets/MUSDB18-7/test/Enda Reilly - Cur An Long Ag Seol.stem.mp4" 

    # run separation
    separate_audio(test_file, model)
