# Training script for Poketrix

import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.generator import Generator
from models.discriminator import Discriminator
from utils.preprocessing import resize_and_normalize, extract_dominant_color

# Hyperparameters
NOISE_DIM = 100
CONDITION_DIM = 10  # Example: 10 types/colors
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

generator = Generator(NOISE_DIM, CONDITION_DIM).to(device)
discriminator = Discriminator(CONDITION_DIM).to(device)

generator_optimizer = Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
discriminator_optimizer = Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

criterion = torch.nn.BCELoss()

# Updated dataset loading with preprocessing
from utils.preprocessing import resize_and_normalize, extract_dominant_color

import pandas as pd

class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        
        # Load CSV
        csv_path = os.path.join(data_dir, "pokedex.csv")
        sprites_dir = os.path.join(data_dir, "sprites")
        
        if not os.path.exists(csv_path) or not os.path.exists(sprites_dir):
            return
            
        df = pd.read_csv(csv_path)
        
        # Build mapping from pokedex_id to (type1, primary_color)
        # Using string mapping to match what's in main.py
        type_mapping = {"Fire": 0, "Water": 1, "Grass": 2, "Electric": 3}
        color_mapping = {"Red": 0, "Blue": 1, "Green": 2, "Yellow": 3}
        
        # Iterate over folders in sprites_dir
        for folder_name in os.listdir(sprites_dir):
            folder_path = os.path.join(sprites_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            
            # The folder name format usually ends with "-<id>" e.g., 0000-Bulbasaur-1
            parts = folder_name.split('-')
            if len(parts) < 3: continue
            
            try:
                pokedex_id = int(parts[-1])
            except ValueError:
                continue
                
            # Find the pokemon in the dataframe
            row = df[df['pokedex_id'] == pokedex_id].iloc[0] if not df[df['pokedex_id'] == pokedex_id].empty else None
            if row is None: continue
            
            p_type = row['type1']
            p_color = row['primary_color']
            
            # To simplify, we only train on ones matching our 4 types/colors 
            # Or we could expand the conditions. For now, try our mapping, ignore others.
            if p_type not in type_mapping or p_color not in color_mapping:
                # Map unknown types/colors to index 0 dynamically as fallback or just skip
                # For this dataset, we will skip or just assign 0 to prevent crash.
                type_idx = type_mapping.get(p_type, 0) # Fallback to 0 (Fire)
                color_idx = color_mapping.get(p_color, 0) # Fallback to 0 (Red)
            else:
                type_idx = type_mapping[p_type]
                color_idx = color_mapping[p_color]
                
            # Collect images from front/normal
            front_normal_dir = os.path.join(folder_path, "front", "normal")
            if os.path.exists(front_normal_dir):
                for img_name in os.listdir(front_normal_dir):
                    if img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(front_normal_dir, img_name))
                        
                        # Create condition vector (10 dims: 4 for types, 4 for colors, 2 extra)
                        condition = torch.zeros(10)
                        condition[type_idx] = 1
                        condition[4 + color_idx] = 1
                        self.labels.append(condition)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        full_path = self.image_paths[idx]
        image = resize_and_normalize(full_path)
        dominant_color = extract_dominant_color(full_path) # We run this but condition is from labels
        image = torch.tensor(image).permute(2, 0, 1) # HWC to CHW

        condition = self.labels[idx]
        return image, condition

def load_dataset():
    """Load and preprocess dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "pokemon_images")
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []
    
    dataset = PokemonDataset(data_dir)
    if len(dataset) == 0:
        return []
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader = load_dataset()

if len(data_loader) == 0:
    print("Warning: No images found in data/images! Please download a Pokémon dataset, place images in backend/data/images, and re-run train.py.")
    # Exit or dummy out to not crash. We'll let it just run 0 epochs.

# Training Loop
for epoch in range(EPOCHS):
    if len(data_loader) == 0:
        break
    
    for i, (real_images, conditions) in enumerate(data_loader):
        real_images = real_images.to(device)
        conditions = conditions.to(device)

        # Train Discriminator
        curr_batch_size = real_images.size(0)
        real_labels = torch.ones(curr_batch_size).to(device)
        fake_labels = torch.zeros(curr_batch_size).to(device)

        # Real images
        real_predictions = discriminator(real_images, conditions).squeeze()
        real_loss = criterion(real_predictions, real_labels)

        # Fake images
        noise = torch.randn(curr_batch_size, NOISE_DIM).to(device)
        fake_images = generator(noise, conditions)
        fake_predictions = discriminator(fake_images.detach(), conditions).squeeze()
        fake_loss = criterion(fake_predictions, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train Generator
        fake_predictions = discriminator(fake_images, conditions).squeeze()
        generator_loss = criterion(fake_predictions, real_labels)

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - D Loss: {discriminator_loss.item():.4f}, G Loss: {generator_loss.item():.4f}")

# Save models
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
torch.save(generator.state_dict(), os.path.join(base_dir, "generator.pth"), _use_new_zipfile_serialization=False)
torch.save(discriminator.state_dict(), os.path.join(base_dir, "discriminator.pth"), _use_new_zipfile_serialization=False)