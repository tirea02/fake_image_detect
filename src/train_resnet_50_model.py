import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from config import TRAIN_DIR, VALID_DIR

if __name__ == "__main__":
    # ✅ Enable CUDA Optimizations
    torch.backends.cudnn.benchmark = True

    # ✅ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load Pretrained ResNet-50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 2)  # Modify final layer for binary classification
    model = model.to(device)

    # ✅ Data Augmentation to Improve Generalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
        transforms.RandomRotation(15),  # Rotate images up to 15 degrees
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Adjust colors
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random small shifts
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random cropping
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ✅ Load Data with Optimized Settings
    batch_size = 64  # Smaller batch size for better generalization
    num_workers = 4  # Parallel data loading
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(VALID_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ✅ Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0002)  # Increased learning rate for faster convergence

    # ✅ Enable AMP for Speed Boost
    scaler = torch.cuda.amp.GradScaler()

    # ✅ Fine-Tune More Layers (Not Just `fc`)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # ✅ Unfreeze last residual block + `fc` layer for deeper learning
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    # ✅ Training Loop (More Epochs for Higher Accuracy)
    num_epochs = 30  # Increased for max accuracy
    best_accuracy = 0.0

    # ✅ Ensure the "models" directory exists
    model_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # ✅ Use AMP for faster training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        # ✅ Validation Accuracy Check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")
        print( f"✅ Epoch {epoch + 1} - Loss: {running_loss:.4f}, Training Acc: {train_acc:.2f}%, Validation Acc: {val_acc:.2f}%")

        # ✅ Save best model only
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_save_path = os.path.join(model_dir, "resnet50_30_64_2.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 Best model saved at: {model_save_path} with Accuracy: {best_accuracy:.2f}%")

    print("🎉 Training complete!")
