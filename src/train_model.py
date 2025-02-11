import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import TRAIN_DIR, VALID_DIR  # Import global dataset paths
from model import DeepCNN  # Import deep CNN model


# Force PyTorch to use GPU (disable fallback to CPU)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"🔥 Running on GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# Load datasets
train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = ImageFolder(VALID_DIR, transform=transform)

# Create DataLoaders
batch_size = 48
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = DeepCNN().to(device)


# Ensure tensors are on GPU
for param in model.parameters():
    assert param.device.type == "cuda", "🚨 Model is still on CPU!"
print(f"✅ Model fully moved to {device}")

# ✅ Debugging: Check if the model and data are on the correct device
print(f"🔍 Model is on: {next(model.parameters()).device}")

# Load a sample batch to check if data is on GPU
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"🖼️ Images device before transfer: {images.device}")  # Should be CPU initially
images = images.to(device)
labels = labels.to(device)
print(f"✅ Images device after transfer: {images.device}")  # Should be CUDA if GPU is working

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
num_epochs = 10
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)  # ✅ Move images & labels to GPU

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_progress.set_postfix(loss=loss.item())  # Show dynamic loss in progress bar

    train_losses.append(running_loss / len(train_loader))

    # Validation Accuracy Calculation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # ✅ Move validation data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    print(
        f"✅ Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")

print("🎉 Training complete!")


# Ensure the "models" directory exists
model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)

# Save the model in the "models" folder
model_save_path = os.path.join(model_dir, "deep_cnn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model successfully saved at: {model_save_path}")

# # Save the trained model  <- wrong
# model_save_path = "deep_cnn_model.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"💾 Model saved as {model_save_path}")

# Plot Loss & Accuracy
plt.figure(figsize=(10, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Time")
plt.legend()

plt.show()
