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

    # ✅ Fine-Tune Deeper Layers (Better AI-Generated Image Detection)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    for param in model.layer3.parameters():
        param.requires_grad = True  # ✅ Unfreeze layer3 for fine-tuning

    for param in model.layer4.parameters():
        param.requires_grad = True  # ✅ Unfreeze layer4 for fine-tuning

    for param in model.fc.parameters():
        param.requires_grad = True  # ✅ Always train the final layer

    # ✅ Data Augmentation (Prevents Overfitting)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip images
        transforms.RandomRotation(15),  # Rotate images slightly
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Color variations
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Shift images slightly
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random zoom
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ✅ Load Data
    batch_size = 64
    num_workers = 4  # Use multiple workers for faster data loading
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(VALID_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ✅ Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)  # ✅ Lower learning rate for stability

    # ✅ Learning Rate Scheduler (Reduces `lr` every 15 epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ✅ Training Loop with Accuracy Tracking
    num_epochs = 50
    best_accuracy = 0.0  # Track best accuracy

    # ✅ Ensure the "models" directory exists
    model_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

            # ✅ Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        scheduler.step()  # ✅ Reduce learning rate every 15 epochs

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

        val_acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Validation Acc: {val_acc:.2f}%")

        # ✅ Save the best model based on validation accuracy
        model_save_path = os.path.join(model_dir, "resnet50_best.pth")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 Best model saved at: {model_save_path} with Accuracy: {best_accuracy:.2f}%")

    print("🎉 Training complete!")
