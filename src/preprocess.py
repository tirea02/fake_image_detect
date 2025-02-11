import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import TRAIN_DIR, VALID_DIR, TEST_DIR  # Import global dataset paths

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets using the global paths
train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = ImageFolder(VALID_DIR, transform=transform)
test_dataset = ImageFolder(TEST_DIR, transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} valid, {len(test_dataset)} test")
