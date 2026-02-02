import os
from tqdm import tqdm
import datetime
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import SingingVoiceDataset
from model import UnetSeparator
from config import CHECKPOINTS_PATH

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# results folder
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_PATH = os.path.join(CHECKPOINTS_PATH, f"training_{formatted_datetime}")




# Train configuration
BATCH_SIZE = 16         # Batch sizes in audio U-Nets are usually 8–16
LEARNING_RATE = 1e-4    # Default Adam LR for image/audio U-Nets
NUM_EPOCHS = 100        # range is 20-100




def train():

    print(f"--- Démarrage de l'entraînement sur {DEVICE} ---")

    # 1. prepare results folder
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # 2. Data

    # init datasets
    print("Dataset loading...")
    train_dataset = SingingVoiceDataset(split='train')
    val_dataset = SingingVoiceDataset(split='validation')

    # DataLoader batching [1, 512, 128] -> [16, 1, 512, 128]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)

    # 3. Model, Loss, Optimizer

    # load model to device
    model = UnetSeparator().to(DEVICE)

    # loss (L'article utilise la norme L1 (Mean Absolute Error))
    criterion = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. history
    history = {
        'train_loss' : [],
        'val_loss' : []
    }
    best_val_loss = float('inf')

    # Main Loop
    for epoch in range(NUM_EPOCHS) : 
        # A. Train

        # switch to train mode (for dropout, batchnorm)
        model.train()
        train_running_loss = 0.0
        num_train_batches = 0

        # progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        # batch loop
        for batch_idx, (x, y) in enumerate(pbar):

            # load x,y to device, x: Mix Spectrogram, y: Vocal Spectrogram
            x = x.to(DEVICE) 
            y = y.to(DEVICE)

            # reset gradients at each batch
            optimizer.zero_grad()

            # forward pass
            prediction = model(x)

            # compute loss
            loss = criterion(prediction, y)

            # compute gradient ∂loss / ∂weight for each parameter (stored in model.param.grad : model.conv1.weight.grad)
            loss.backward()

            # update weights
            optimizer.step()

            # stats 
            train_running_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix({'loss' : f"{loss.item():.4f}"})

        epoch_train_loss = train_running_loss / num_train_batches
        history['train_loss'].append(epoch_train_loss)   

        # B. Validation

        # inference mode
        model.eval()
        val_running_loss = 0.0
        num_val_batches = 0

        with torch.no_grad() : # deactivate gradient computations

            # validation batch loop
            for x_val, y_val in val_loader :

                # load x, y
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)

                # forward
                pred_val = model(x_val)

                # loss
                loss_val = criterion(pred_val, y_val)

                # update total loss
                val_running_loss += loss_val.item()
                num_val_batches += 1

        epoch_val_loss = val_running_loss / num_val_batches if num_val_batches > 0 else 0
        history['val_loss'].append(epoch_val_loss)

        # display 
        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss {epoch_val_loss:.4f}")

        # C. save models and history

        # save last model
        last_model_path = os.path.join(RESULTS_PATH, "voice_separator_last.pt")
        torch.save(model.state_dict(), last_model_path)

        # save best model
        if epoch_val_loss < best_val_loss : 
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(RESULTS_PATH, "voice_separator_best.pt")
            torch.save(model.state_dict(), best_model_path)

        # save loss history
        history_file = os.path.join(RESULTS_PATH, "training_history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        # save loss plot
        history_file = os.path.join(RESULTS_PATH, "training_history.json")
        save_training_history(history_file)

    print("Training Complete !")




def save_training_history(json_path):
    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.title('Courbes d\'apprentissage U-Net')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(json_path), "training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()




# Run main 
if __name__ == "__main__":

    # rin with python -m implementation.train

    # run training
    train()

    

        




