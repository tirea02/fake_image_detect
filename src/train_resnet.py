# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from torchvision.models import resnet18, ResNet18_Weights
# from tqdm import tqdm
# from config import TRAIN_DIR, VALID_DIR
#
# # ✅ Fix for Windows multiprocessing issue
# if __name__ == "__main__":
#     # ✅ Use GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # ✅ Load Pretrained ResNet-18
#     model = resnet18(weights=ResNet18_Weights.DEFAULT)  # ✅ Updated from deprecated `pretrained=True`
#     model.fc = nn.Linear(512, 2)  # Modify final layer for binary classification
#     model = model.to(device)
#
#     # ✅ Define Transformations
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     # ✅ Load Data
#     train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
#     val_dataset = ImageFolder(VALID_DIR, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)  # ✅ Set num_workers=0
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
#
#     # ✅ Loss Function & Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # Train only last layer
#
#     # ✅ Training Loop
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
#
#         for images, labels in train_progress:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             train_progress.set_postfix(loss=loss.item())
#
#         print(f"✅ Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")
#
#     # Ensure the "models" directory exists
#     model_dir = os.path.join(os.path.dirname(__file__), "../models")
#     # os.makedirs(model_dir, exist_ok=True)
#
#     # Save the model in the "models" folder
#     model_save_path = os.path.join(model_dir, "deep_cnn_model.pth")
#     torch.save(model.state_dict(), model_save_path)
#     print(f"✅ Model successfully saved at: {model_save_path}")
#
#     # # ✅ Save Model
#     # model_save_path = os.path.join(os.path.dirname(__file__), "models", "resnet18.pth")
#     # torch.save(model.state_dict(), model_save_path)
#     # print(f"✅ ResNet model saved at: {model_save_path}")



import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from config import TRAIN_DIR, VALID_DIR

# ✅ Fix for Windows multiprocessing issue
if __name__ == "__main__":
    # ✅ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load Pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 2)  # Modify final layer for binary classification
    model = model.to(device)

    # ✅ Define Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ✅ Load Data
    batch_size = 64
    num_workers = 2  # Use 2 for speed, adjust as needed
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(VALID_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ✅ Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # Train only last layer

    # ✅ Training Loop with Accuracy Tracking
    num_epochs = 10
    best_accuracy = 0.0  # Track best accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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

        # ✅ Save best model
        model_dir = os.path.join(os.path.dirname(__file__), "../models")
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "resnet18_best_2.pth")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 Best model saved at: {model_save_path} with Accuracy: {best_accuracy:.2f}%")

    print("🎉 Training complete!")
